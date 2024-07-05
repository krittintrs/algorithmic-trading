import requests
import pandas as pd
import time
import numpy as np

def fetch_live_data():
    url = 'https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT'
    response = requests.get(url)
    data = response.json()
    return float(data['price'])

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
    return df
    
def preprocess_data(df):
    df['Returns'] = df['Close'].pct_change()
    # df.dropna(inplace=True)
    return df

# Function to update the DataFrame with new live data
def update_data(df):
    price = fetch_live_data()
    timestamp = pd.Timestamp.now()
    new_data = pd.DataFrame({'Date': [timestamp], 'Close': [price]})
    df = pd.concat([df, new_data])
    df.set_index('Date', inplace=True)
    df = preprocess_data(df)
    return df

from agent_ma import MovingAverageCrossoverAgent
from agent_dummy import DummyAgent

# Initialize DataFrame to store live data
df = pd.DataFrame(columns=['Date', 'Close'])

# Initialize agents
mac_agent = MovingAverageCrossoverAgent()
dummy_agent = DummyAgent()

# Function to update agents and print portfolio values
def update_agents(df, t):
    df = update_data(df)
    current_price = df['Close'].iloc[-1]
    mac_agent.trade(df)
    dummy_agent.trade(df)
    print(f"MAC Agent Portfolio Value: {mac_agent.get_portfolio_value(current_price)}")
    print(f"Dummy Agent Portfolio Value: {dummy_agent.get_portfolio_value(current_price)}")
    print()

    return df

interval = 0.5
t = 0
while True:
    df = update_agents(df, t)
    time.sleep(interval)  # Fetch data and trade every minute
    t += 1