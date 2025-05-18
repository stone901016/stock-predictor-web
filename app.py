from flask import Flask, request, jsonify, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/autocomplete')
def autocomplete():
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])
    url = f'https://query2.finance.yahoo.com/v1/finance/search?q={q}'
    try:
        res = requests.get(url)
        items = res.json().get('quotes', [])
    except Exception:
        return jsonify([])
    results = []
    for it in items:
        sym = it.get('symbol')
        name = it.get('shortname') or it.get('longname') or it.get('exchange')
        if sym:
            results.append({'symbol': sym, 'name': name})
    return jsonify(results)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    symbol = data.get('symbol')
    date_mode = data.get('date_mode')
    interval = data.get('interval')

    # 日期範圍設定
    if date_mode == 'range':
        start_date = data.get('start_date')
        end_date = data.get('end_date')
    elif date_mode == 'year':
        years = int(data.get('years', 1))
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=years)).strftime('%Y-%m-%d')
    else:
        start_date = '1990-01-01'
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if df.empty:
            return jsonify({'error': '找不到資料，請確認股票代號與日期'}), 400

        # 平整欄位名稱
        flat_cols = []
        for col in df.columns:
            flat_cols.append(col[0] if isinstance(col, tuple) else col)
        df.columns = [str(c).strip() for c in flat_cols]

        # 處理缺少 Adj Close
        if 'Adj Close' not in df.columns:
            cands = [c for c in df.columns if 'close' in c.lower()]
            if cands:
                df['Adj Close'] = df[cands[0]]
            else:
                return jsonify({'error': f"無可用的 'Adj Close'，目前欄位：{df.columns.tolist()}"}), 400

        # 計算報酬與年份
        df['Return'] = df['Adj Close'].pct_change()
        df['Year'] = df.index.year

        # 每年累積報酬率與平均
        annual_returns = df.groupby('Year')['Return'] \
                         .apply(lambda r: (r.add(1).prod() - 1))
        avg_return = float(annual_returns.sum() / len(annual_returns))

        # 年度波動率
        volatility = (df.groupby('Year')['Return'].std() * np.sqrt(252)).round(4).to_dict()

        # 最大回測
        cumulative = (1 + df['Return'].fillna(0)).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = float(drawdown.min().round(4))

        # Sharpe Ratio
        sharpe = float(((df['Return'].mean() * 252) / (df['Return'].std() * np.sqrt(252))) \
                       if df['Return'].std() else 0)

        # 選擇市場指數
        sym = symbol.upper()
        if sym.endswith('.TW'):
            mkt_ix = '^TWII'
        elif sym.endswith(('.KS', '.KQ')):
            mkt_ix = '^KS11'
        elif sym.endswith('.T'):
            mkt_ix = '^N225'
        elif sym.endswith('.HK'):
            mkt_ix = '^HSI'
        else:
            mkt_ix = '^GSPC'

        # 下載市場資料
        mkt = yf.download(mkt_ix, start=start_date, end=end_date, interval=interval)
        flat_mkt = []
        for col in mkt.columns:
            flat_mkt.append(col[0] if isinstance(col, tuple) else col)
        mkt.columns = [str(c).strip() for c in flat_mkt]

        retcol = 'Adj Close' if 'Adj Close' in mkt.columns else ('Close' if 'Close' in mkt.columns else None)
        if not retcol:
            return jsonify({'error': f"市場指數 {mkt_ix} 缺少收盤價，目前欄位：{mkt.columns.tolist()}"}), 400
        mkt['Return'] = mkt[retcol].pct_change()

        # 計算 Alpha & Beta
        idx = df['Return'].dropna().index.intersection(mkt['Return'].dropna().index)
        cov = np.cov(df.loc[idx, 'Return'], mkt.loc[idx, 'Return'])
        beta = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] else None
        alpha = float((df['Return'].mean() - beta * mkt['Return'].mean()) * 252) if beta else None

        # 繪圖設定
        plt.rcParams.update({'font.size': 12})
        fig, ax = plt.subplots(figsize=(10, 4))
        cumulative.plot(ax=ax)
        ax.set_title(f'{symbol} NAV', fontsize=16)
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Cumulative Return (NAV)', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(True)

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png')
        plt.close()
        nav_img = base64.b64encode(buf.getvalue()).decode()

        # 回傳結果
        return jsonify({
            'volatility': volatility,
            'annual_returns': {int(y): round(r, 4) for y, r in annual_returns.items()},
            'avg_return': round(avg_return, 4),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': round(sharpe, 4),
            'alpha': round(alpha, 4) if alpha else None,
            'beta': round(beta, 4) if beta else None,
            'nav_img': nav_img
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)
