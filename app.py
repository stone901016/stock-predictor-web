from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocomplete')
def autocomplete():
    query = request.args.get('q', '').upper()
    symbols = ['AAPL', 'MSFT', 'TSLA', 'GOOG', 'AMZN', '2330.TW', '2303.TW']
    return jsonify([s for s in symbols if query in s])

@app.route('/analysis', methods=['POST'])
def analysis():
    data = request.json
    symbol = data['symbol']
    start_date = data['start_date']
    end_date = data['end_date']
    interval = data['interval']

    df = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    if df.empty:
        return jsonify({'error': '無法下載資料'})

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

    # NAV 圖
    nav = (1 + df['Return'].fillna(0)).cumprod()
    fig, ax = plt.subplots(figsize=(10, 4))
    nav.plot(ax=ax)
    plt.title(f'{symbol} NAV 淨值曲線')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    nav_img = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # RSI 圖
    rsi = RSIIndicator(df['Adj Close']).rsi()
    fig, ax = plt.subplots(figsize=(10, 3))
    rsi.plot(ax=ax)
    plt.axhline(70, color='r', linestyle='--')
    plt.axhline(30, color='g', linestyle='--')
    plt.title(f'{symbol} RSI 指標')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    rsi_img = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # MACD 圖
    macd = MACD(df['Adj Close'])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, macd.macd(), label='MACD')
    ax.plot(df.index, macd.macd_signal(), label='Signal')
    ax.legend()
    plt.title(f'{symbol} MACD 指標')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    macd_img = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # 布林帶圖
    bb = BollingerBands(df['Adj Close'])
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df['Adj Close'], label='收盤價')
    ax.plot(df.index, bb.bollinger_hband(), linestyle='--', label='上軌')
    ax.plot(df.index, bb.bollinger_lband(), linestyle='--', label='下軌')
    ax.legend()
    plt.title(f'{symbol} 布林通道')
    buf = BytesIO()
    plt.savefig(buf, format='png')
    bb_img = base64.b64encode(buf.getvalue()).decode()
    plt.close()

    # 匯出 Excel（分析報表）
    excel_buf = BytesIO()
    excel_df = pd.DataFrame({
        'Year': volatility.index,
        'Volatility': volatility.values,
        'Average Return': avg_return.values
    })
    excel_df['Max Drawdown'] = max_drawdown
    excel_df['Sharpe Ratio'] = sharpe_ratio
    excel_df['Alpha'] = alpha
    excel_df['Beta'] = beta
    excel_df.to_excel(excel_buf, index=False)
    excel_buf.seek(0)
    excel_b64 = base64.b64encode(excel_buf.read()).decode()

    return jsonify({
        'volatility': volatility.to_dict(),
        'average_return': avg_return.to_dict(),
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'alpha': alpha,
        'beta': beta,
        'nav_img': nav_img,
        'rsi_img': rsi_img,
        'macd_img': macd_img,
        'bb_img': bb_img,
        'excel_data': excel_b64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
