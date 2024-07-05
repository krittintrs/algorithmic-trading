import requests
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def fetch_historical_data(symbol, interval, limit=1000):
    """
    interval:
    1m, 3m, 5m, 15m, 30m
    1h, 2h, 4h, 6h, 8h, 12h,
    1d, 3d, 1w, 1M
    """
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    df = df.rename(columns=lambda x: x.capitalize())
    return df

class XGBoostAgent:
    def __init__(self):
        self.name = "XGBoostTrader"
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.cash = 100000  # Starting cash in USD
        self.holdings = 0
        self.model = None
        self.is_trained = False
        self.window_size = 4  # Adjust this as per your requirement
        self.buffered = fetch_historical_data('BTCUSDT', '1m').tail(10)

    def prepare_data(self, data):
        # Ensure data has necessary columns
        if 'Close' not in data.columns:
            raise ValueError("Input data does not contain 'Close' column.")

        # print('before prepare: ')
        # print(data)

        # Calculate additional features
        data['Price_Change'] = data['Close'].pct_change()

        if len(data) >= 10:
            data['Volatility'] = data['Close'].rolling(window=10).std()
            data['Momentum'] = data['Close'] - data['Close'].shift(10)
        else:
            data['Volatility'] = data['Close'].rolling(window=len(data)).std()
            data['Momentum'] = data['Close'] - data['Close'].shift(len(data))

        # print('after calculating features: ')
        # print(data)

        data['Volatility'].fillna(data['Volatility'].mean(), inplace=True)
        data['Momentum'].fillna(data['Momentum'].mean(), inplace=True)

        # print('after NaNs: ')
        # print(data)

        # Check if data is empty after preparation
        if len(data) == 0:
            raise ValueError("Data is empty after preparation.")

        return data
    
    def generate_labels(self, data):
        data['Signal'] = np.where(data['Close'] > data['Close'].shift(1), 1, np.where(data['Close'] < data['Close'].shift(1), 2, 0))
        return data

    def train_model(self, data):
        data = self.prepare_data(data)
        data = self.generate_labels(data)

        features = data[['Price_Change', 'Volatility', 'Momentum']]
        labels = data['Signal']

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        params = {
            'objective': 'multi:softmax',  # Adjusted for multi-class classification
            'num_class': 3,  # Number of classes (0, 1, 2)
            'max_depth': 5,
            'eta': 0.1,
            'eval_metric': 'mlogloss'
        }

        self.model = xgb.train(params, dtrain, num_boost_round=100)

        y_pred = self.model.predict(dtest)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model trained with accuracy: {accuracy}")

        self.is_trained = True

    def generate_signal(self, data):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Please train the model before generating signals.")

        # print('before prepare: ')
        # print(data)

        data = self.prepare_data(data)

        # print('after prepare: ')
        # print(data)

        if len(data) == 0:
            raise ValueError("Data is empty after preparation.")

        features = data[['Price_Change', 'Volatility', 'Momentum']]
        dfeatures = xgb.DMatrix(features)

        if dfeatures.num_row() == 0:
            raise ValueError("DMatrix is empty. Check if features are correctly prepared.")

        signal = self.model.predict(dfeatures)[-1]  # Predict signal for the last row
        return int(signal)  # Ensure signal is integer for consistency


    def trade(self, data):
        if len(data) < self.window_size:
            print(f"{pd.Timestamp.now()}: {self.name} Not enough data to make a decision.")
            # print(data)
            # print('add historical data')
            
            # Align columns of self.buffered and data before concatenating
            common_columns = self.buffered.columns.intersection(data.columns)
            self.buffered = self.buffered[common_columns]
            data = pd.concat([self.buffered, data], ignore_index=True)

        signal = self.generate_signal(data)
        current_price = data['Close'].iloc[-1]  # Get the latest closing price

        if signal == 1 and self.position != 1:
            if self.cash > 0:
                self.holdings = self.cash / current_price
                self.cash = 0
                self.position = 1
                print(f"{pd.Timestamp.now()}: {self.name} Buy at {current_price}")
        elif signal == 2 and self.position != -1:
            if self.holdings > 0:
                self.cash = self.holdings * current_price
                self.holdings = 0
                self.position = -1
                print(f"{pd.Timestamp.now()}: {self.name} Sell at {current_price}")
        else:
            print(f"{pd.Timestamp.now()}: {self.name} Hold")

    def get_portfolio_value(self, current_price):
        return self.cash + (self.holdings * current_price)
