import pandas as pd
import numpy as np

def ma_signals(data, short=5, long=20):
    data = data.copy()
    data['MA_short'] = data['Close'].rolling(short).mean()
    data['MA_long'] = data['Close'].rolling(long).mean()
    data['MA_Cross'] = (data['MA_short'] > data['MA_long']).astype(int)
    return data

def rsi_signal(data, window=14, overbought=70, oversold=30):
    data = data.copy()
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    rs = roll_up / roll_down
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_signal'] = 0
    data.loc[data['RSI'] > overbought, 'RSI_signal'] = -1
    data.loc[data['RSI'] < oversold, 'RSI_signal'] = 1
    return data

def macd_signal(data, fast=12, slow=26, signal=9):
    data = data.copy()
    data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
    data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
    data['MACD'] = data['EMA_fast'] - data['EMA_slow']
    data['MACD_signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    data['MACD_Cross'] = (data['MACD_hist'] > 0).astype(int)
    return data

def bollinger_signal(data, window=20, num_std=2):
    data = data.copy()
    data['MA'] = data['Close'].rolling(window).mean()
    data['STD'] = data['Close'].rolling(window).std()
    data['Upper'] = data['MA'] + num_std * data['STD']
    data['Lower'] = data['MA'] - num_std * data['STD']
    data['BOLL_Break'] = 0
    data.loc[data['Close'] > data['Upper'], 'BOLL_Break'] = 1
    data.loc[data['Close'] < data['Lower'], 'BOLL_Break'] = -1
    return data

def volume_breakout_signal(data, window=10, threshold=2):
    data = data.copy()
    data['VOL_Mean'] = data['Volume'].rolling(window).mean()
    data['VOL_Break'] = (data['Volume'] > data['VOL_Mean'] * threshold).astype(int)
    return data

# 可继续扩展更多信号，如题材/行业动量（留接口）、涨停、主力流入等
