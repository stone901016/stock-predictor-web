from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    symbol = data.get('symbol')
    date_mode = data.get('date_mode')
    interval = data.get('interval')

    # 設定日期範圍
    if date_mode == 'range':
        start_date = data.get('start_date')
        end_date = data.get('end_date')
    elif date_mode == 'year':
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        years = int(data.get('years', 1))
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    else:  # all
        start_date = '1990-01-01'
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if df.empty:
            return jsonify({'error': '找不到資料，請確認股票代號與日期'}), 400

        # 清理並平整欄位名稱
        flat_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                flat_cols.append(col[0])
            else:
                flat_cols.append(col)
        df.columns = [str(c).strip() for c in flat_cols]

        # 處理缺少 Adj Close
        if 'Adj Close' not in df.columns:
            candidates = [col for col in df.columns if 'close' in col.lower()]
            if candidates:
                df['Adj Close'] = df[candidates[0]]
            else:
                return jsonify({'error': f"資料欄位中沒有可用的 'Adj Close'，目前欄位有：{df.columns.tolist()}"}), 400

        # 計算報酬與年份
        df['Return'] = df['Adj Close'].pct_change()
        df['Year'] = df.index.year

        # 每年累積報酬率
        annual_returns = df.groupby('Year')['Return'] \
            .apply(lambda r: (r.add(1).prod() - 1))
        # 年平均累積報酬率
        avg_return = float(annual_returns.sum() / len(annual_returns))

        # 年度波動率
        volatility = (df.groupby('Year')['Return'].std() * np.sqrt(252)).round(4).to_dict()

        # 最大回測
        cumulative = (1 + df['Return'].fillna(0)).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = float(drawdown.min().round(4))

        # Sharpe Ratio
        sharpe_ratio = float(((df['Return'].mean() * 252) / (df['Return'].std() * np.sqrt(252))) \
                             if df['Return'].std() != 0 else 0)

        # 選擇市場指數
        sym = symbol.upper()
        if sym.endswith('.TW'):
            market_ix = '^TWII'
        elif sym.endswith(('.KS', '.KQ')):
            market_ix = '^KS11'
        elif sym.endswith('.T'):
            market_ix = '^N225'
        elif sym.endswith('.HK'):
            market_ix = '^HSI'
        else:
            market_ix = '^GSPC'

        mkt = yf.download(market_ix, start=start_date, end=end_date, interval=interval)
        # 清理並平整市場指數欄位
        flat_mkt_cols = []
        for col in mkt.columns:
            if isinstance(col, tuple):
                flat_mkt_cols.append(col[0])
            else:
                flat_mkt_cols.append(col)
        mkt.columns = [str(c).strip() for c in flat_mkt_cols]

        mkt_return_col = 'Adj Close' if 'Adj Close' in mkt.columns else ('Close' if 'Close' in mkt.columns else None)
        if not mkt_return_col:
            return jsonify({'error': f"市場指數 {market_ix} 缺少收盤價欄位，目前欄位有：{mkt.columns.tolist()}"}), 400
        mkt['Return'] = mkt[mkt_return_col].pct_change()

        idx = df['Return'].dropna().index.intersection(mkt['Return'].dropna().index)
        cov = np.cov(df.loc[idx, 'Return'], mkt.loc[idx, 'Return'])
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] != 0 else None
        alpha = float((df['Return'].mean() - beta * mkt['Return'].mean()) * 252) if beta is not None else None

        # 繪製 NAV 圖並移除邊框
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(10, 4))
        cumulative.plot(ax=ax)
        ax.set_title(f'{symbol} NAV', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Return (NAV)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        # 移除邊框
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_frame_on(False)
        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        nav_img = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'volatility': volatility,
            'annual_returns': {int(year): round(val, 4) for year, val in annual_returns.items()},
            'avg_return': round(avg_return, 4),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': round(sharpe_ratio, 4),
            'alpha': round(alpha, 4) if alpha is not None else None,
            'beta': round(beta, 4) if beta is not None else None,
            'nav_img': nav_img
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
