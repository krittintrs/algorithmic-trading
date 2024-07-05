import numpy as np
import pandas as pd

class MovingAverageCrossoverAgent:
    def __init__(self, short_window=2, long_window=5):
        self.name = "MAC"
        self.position = 0  # 0: No position, 1: Long, -1: Short
        self.cash = 100000  # Starting cash in USD
        self.holdings = 0
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        # Randomly choose an action: 0 (Hold), 1 (Buy), 2 (Sell)
        data['Short_MA'] = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        data['Long_MA'] = data['Close'].rolling(window=self.long_window, min_periods=1).mean()

        # print('MA short: ' + str(data['Short_MA'].iloc[-1]) + ' / MA long: ' + str(data['Long_MA'].iloc[-1]))
        # print('self pos: ' + str(self.position))

        if data['Short_MA'].iloc[-1] > data['Long_MA'].iloc[-1] and self.position != 1:
            return 1  # Buy signal
        elif data['Short_MA'].iloc[-1] < data['Long_MA'].iloc[-1] and self.position != -1:
            return 2  # Sell signal
        else:
            return 0  # Hold signal

    def trade(self, data):
        signal = self.generate_signals(data)
        price = data['Close'].iloc[-1]
        if signal == 1 and self.position != 1:
            if self.cash > 0:
                self.holdings = self.cash / price
                self.cash = 0
                self.position = 1
                print(f"{pd.Timestamp.now()}: {self.name} Buy at {price}")
        elif signal == 2 and self.position != -1:
            if self.holdings > 0:
                self.cash = self.holdings * price
                self.holdings = 0
                self.position = -1
                print(f"{pd.Timestamp.now()}: {self.name} Sell at {price}")
        else:
            print(f"{pd.Timestamp.now()}: {self.name} Hold")

    def get_portfolio_value(self, current_price):
        return self.cash + (self.holdings * current_price)