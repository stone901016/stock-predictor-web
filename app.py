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
    symbol = data['symbol']
    date_mode = data['date_mode']
    interval = data['interval']
    start_date, end_date = None, None

    if date_mode == 'range':
        start_date = data['start_date']
        end_date = data['end_date']
    elif date_mode == 'year':
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
        start_date = (pd.Timestamp.today() - pd.DateOffset(years=int(data['years']))).strftime('%Y-%m-%d')
    elif date_mode == 'all':
        start_date = '1990-01-01'
        end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

    try:
        df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
        if df.empty:
            return jsonify({'error': '找不到資料，請確認股票代號與日期'}), 400

        df['Return'] = df['Adj Close'].pct_change()
        df['Year'] = df.index.year

        volatility = df.groupby('Year')['Return'].std() * np.sqrt(252)
        avg_return = df.groupby('Year')['Return'].mean() * 252

        cumulative = (1 + df['Return'].fillna(0)).cumprod()
        peak = cumulative.cummax()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        sharpe_ratio = (df['Return'].mean() * 252) / (df['Return'].std() * np.sqrt(252))

        market = yf.download('^GSPC', start=start_date, end=end_date, interval=interval)
        market['Return'] = market['Adj Close'].pct_change()
        common_index = df['Return'].dropna().index.intersection(market['Return'].dropna().index)
        cov = np.cov(df.loc[common_index, 'Return'], market.loc[common_index, 'Return'])
        beta = cov[0, 1] / cov[1, 1] if cov[1, 1] != 0 else None
        alpha = (df['Return'].mean() - beta * market['Return'].mean()) * 252 if beta else None

        fig, ax = plt.subplots(figsize=(10, 4))
        cumulative.plot(ax=ax)
        ax.set_title(f'{symbol} NAV 淨值曲線')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        nav_img = base64.b64encode(buf.getvalue()).decode()

        return jsonify({
            'volatility': volatility.round(4).to_dict(),
            'avg_return': avg_return.round(4).to_dict(),
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'alpha': alpha,
            'beta': beta,
            'nav_img': nav_img
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)

