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

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data['symbol']
    start_date = data['start_date']
    end_date = data['end_date']
    forecast_days = int(data['forecast_days'])
    model_type = data['model_type']

    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        return jsonify({'error': '無法下載資料'})

    close_prices = df[['Adj Close']].values if 'Adj Close' in df.columns else df[['Close']].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(close_prices)

    X, y = [], []
    for i in range(60, len(scaled)):
        X.append(scaled[i-60:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    if model_type == 'lstm':
        pred_scaled = []
        input_seq = scaled[-60:]
        for _ in range(forecast_days):
            pred = model.predict(input_seq.reshape(1, 60, 1), verbose=0)[0][0]
            pred_scaled.append(pred)
            input_seq = np.append(input_seq[1:], [[pred]], axis=0)

        pred_actual = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

        fig, ax = plt.subplots()
        ax.plot(df.index, df['Adj Close'], label='歷史價格')
        ax.plot(future_dates, pred_actual, label='LSTM 預測')
        ax.legend()
        plt.title(f'{symbol} LSTM 預測未來 {forecast_days} 天')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        predict_img = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        train_pred = model.predict(X, verbose=0)
        rmse = float(np.sqrt(mean_squared_error(scaler.inverse_transform(y), scaler.inverse_transform(train_pred))))
        mape = float(mean_absolute_percentage_error(scaler.inverse_transform(y), scaler.inverse_transform(train_pred)))

        return jsonify({
            'predict_img': predict_img,
            'rmse': rmse,
            'mape': mape
        })

    elif model_type == 'random':
        last = df['Adj Close'].iloc[-1]
        returns = df['Adj Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        simulated = [last]
        for _ in range(forecast_days):
            simulated.append(simulated[-1] * (1 + np.random.normal(mu, sigma)))
        simulated = simulated[1:]
        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)

        fig, ax = plt.subplots()
        ax.plot(df.index, df['Adj Close'], label='歷史價格')
        ax.plot(future_dates, simulated, label='隨機漫步預測')
        ax.legend()
        plt.title(f'{symbol} 隨機漫步預測')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        predict_img = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        return jsonify({ 'predict_img': predict_img, 'rmse': None, 'mape': None })

    elif model_type == 'compare':
        # LSTM 部分同上
        pred_scaled = []
        input_seq = scaled[-60:]
        for _ in range(forecast_days):
            pred = model.predict(input_seq.reshape(1, 60, 1), verbose=0)[0][0]
            pred_scaled.append(pred)
            input_seq = np.append(input_seq[1:], [[pred]], axis=0)
        pred_lstm = scaler.inverse_transform(np.array(pred_scaled).reshape(-1, 1)).flatten()

        last = df['Adj Close'].iloc[-1]
        returns = df['Adj Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        pred_random = [last]
        for _ in range(forecast_days):
            pred_random.append(pred_random[-1] * (1 + np.random.normal(mu, sigma)))
        pred_random = pred_random[1:]

        future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        fig, ax = plt.subplots()
        ax.plot(future_dates, pred_lstm, label='LSTM 預測')
        ax.plot(future_dates, pred_random, label='隨機漫步')
        ax.legend()
        plt.title(f'{symbol} LSTM vs Random 比較')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        predict_img = base64.b64encode(buf.getvalue()).decode()
        plt.close()

        train_pred = model.predict(X, verbose=0)
        rmse = float(np.sqrt(mean_squared_error(scaler.inverse_transform(y), scaler.inverse_transform(train_pred))))
        mape = float(mean_absolute_percentage_error(scaler.inverse_transform(y), scaler.inverse_transform(train_pred)))

        return jsonify({
            'predict_img': predict_img,
            'lstm_rmse': rmse,
            'lstm_mape': mape
        })

    return jsonify({ 'error': '未支援的模型模式' })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
