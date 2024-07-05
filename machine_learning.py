import pandas as pd
import numpy as np
from simulation import fetch_historical_data
from pycaret.classification import setup, compare_models, predict_model

class MLTuner:
    def __init__(self):
        self.name = "MLTrader"
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.cash = 100000  # Starting cash in USD
        self.holdings = 0
        self.model = None
        self.is_trained = False
        
    def prepare_data(self, data):
        data['Price_Change'] = data['Close'].pct_change()
        data['Volatility'] = data['Close'].rolling(window=10).std()
        data['Momentum'] = data['Close'] - data['Close'].shift(10)
        data.dropna(inplace=True)
        return data

    def generate_labels(self, data):
        data['Signal'] = 0
        data['Signal'] = np.where(data['Close'].shift(1) < data['Close'], 1, -1)
        return data

    def train_model(self, data):
        data = self.prepare_data(data)
        data = self.generate_labels(data)

        print('Start comparing models...')

        # Use PyCaret to setup and compare models
        exp_clf = setup(data=data, target='Signal', verbose=False)
        best_model = compare_models()

        self.model = best_model
        self.is_trained = True

    def generate_signals(self, data):
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Please train the model before generating signals.")

        data = self.prepare_data(data)
        features = data[['Price_Change', 'Volatility', 'Momentum']]
        prediction = predict_model(self.model, data=features.iloc[-1:])
        signal = prediction['Label'].iloc[-1]

        return signal

    def trade(self, data):
        if len(data) < 11:  # Ensure there's enough data to calculate features and generate signals
            print(f"{pd.Timestamp.now()}: {self.name} Not enough data to make a decision.")
            return

        signal = self.generate_signals(data)
        price = data['Close'].iloc[-1]

        if signal == 1 and self.position != 1:
            if self.cash > 0:
                self.holdings = self.cash / price
                self.cash = 0
                self.position = 1
                print(f"{pd.Timestamp.now()}: {self.name} Buy at {price}")
        elif signal == -1 and self.position != -1:
            if self.holdings > 0:
                self.cash = self.holdings * price
                self.holdings = 0
                self.position = -1
                print(f"{pd.Timestamp.now()}: {self.name} Sell at {price}")
        else:
            print(f"{pd.Timestamp.now()}: {self.name} Hold")

    def get_portfolio_value(self, current_price):
        return self.cash + (self.holdings * current_price)

# Example usage
if __name__ == "__main__":
    # Fetching historical data
    agent = MLTuner()
    data = fetch_historical_data('BTCUSDT', '1m')

    # Train the model with historical data
    agent.train_model(data)
    
    # Top 3 Models
    #                                     Model  Accuracy     AUC  Recall   Prec.  \
    # rf               Random Forest Classifier    0.9971  0.9991  0.9971  0.9972
    # ada                  Ada Boost Classifier    0.9971  0.9986  0.9971  0.9972
    # xgboost         Extreme Gradient Boosting    0.9971  0.9970  0.9971  0.9972
