from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data['symbol']
    start_date = data['start_date']
    end_date = data['end_date']
    forecast_days = int(data.get('forecast_days', 30))
    model_type = data.get('model_type', 'lstm')

    df = yf.download(symbol, start=start_date, end=end_date)
    if df.empty:
        return jsonify({'error': '資料下載失敗'})

    if model_type == 'compare':
        close_data = df[['Adj Close']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        X, y = [], []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])

        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X, y, epochs=10, batch_size=32, verbose=0)

        last_sequence = scaled_data[-60:]
        lstm_predictions = []
        current_input = last_sequence.copy()
        for _ in range(forecast_days):
            input_seq = current_input[-60:].reshape(1, 60, 1)
            next_pred = model.predict(input_seq, verbose=0)[0][0]
            lstm_predictions.append(next_pred)
            current_input = np.append(current_input, [[next_pred]], axis=0)

        lstm_predictions = scaler.inverse_transform(np.array(lstm_predictions).reshape(-1, 1)).flatten()
        train_preds = model.predict(X, verbose=0)
        train_preds = scaler.inverse_transform(train_preds)
        actual = scaler.inverse_transform(y.reshape(-1, 1))
        lstm_rmse = float(np.sqrt(mean_squared_error(actual, train_preds)))
        lstm_mape = float(mean_absolute_percentage_error(actual, train_preds))

        last_price = df['Adj Close'].iloc[-1]
        returns = df['Adj Close'].pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        random_predictions = [last_price]
        for _ in range(forecast_days):
            random_predictions.append(random_predictions[-1] * (1 + np.random.normal(mu, sigma)))
        random_predictions = random_predictions[1:]

        dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(dates, lstm_predictions, label='LSTM 預測')
        ax.plot(dates, random_predictions, label='隨機預測')
        plt.title(f"{symbol} LSTM vs Random 預測比較")
        plt.xlabel("日期")
        plt.ylabel("價格")
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        predict_img = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return jsonify({
            'predict_img': predict_img,
            'lstm_rmse': lstm_rmse,
            'lstm_mape': lstm_mape
        })

    return jsonify({'error': '未支援的模型模式'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
