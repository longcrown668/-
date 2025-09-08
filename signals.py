import pandas as pd
import numpy as np

def ma_signals(data, short=5, long=20):
    data = data.copy()
    data['MA_short'] = data['Close'].rolling(short).mean()
    data['MA_long'] = data['Close'].rolling(long).mean()
    # 确保索引对齐
    ma_short_aligned, ma_long_aligned = data['MA_short'].align(data['MA_long'], axis=0)
    data['MA_Cross'] = (ma_short_aligned > ma_long_aligned).astype(int)
    return data.dropna()

def rsi_signal(data, window=14, overbought=70, oversold=30):
    data = data.copy()
    delta = data['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()
    # 避免除零错误
    rs = roll_up / (roll_down + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI_signal'] = 0
    data.loc[data['RSI'] > overbought, 'RSI_signal'] = -1
    data.loc[data['RSI'] < oversold, 'RSI_signal'] = 1
    return data.dropna()

def macd_signal(data, fast=12, slow=26, signal=9):
    data = data.copy()
    data['EMA_fast'] = data['Close'].ewm(span=fast, adjust=False).mean()
    data['EMA_slow'] = data['Close'].ewm(span=slow, adjust=False).mean()
    # 确保索引对齐
    ema_fast_aligned, ema_slow_aligned = data['EMA_fast'].align(data['EMA_slow'], axis=0)
    data['MACD'] = ema_fast_aligned - ema_slow_aligned
    data['MACD_signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
    macd_aligned, macd_signal_aligned = data['MACD'].align(data['MACD_signal'], axis=0)
    data['MACD_hist'] = macd_aligned - macd_signal_aligned
    data['MACD_Cross'] = (data['MACD_hist'] > 0).astype(int)
    return data.dropna()

def bollinger_signal(data, window=20, num_std=2):
    data = data.copy()
    data['MA'] = data['Close'].rolling(window).mean()
    data['STD'] = data['Close'].rolling(window).std()
    # 确保索引对齐
    ma_aligned, std_aligned = data['MA'].align(data['STD'], axis=0)
    data['Upper'] = ma_aligned + num_std * std_aligned
    data['Lower'] = ma_aligned - num_std * std_aligned
    data['BOLL_Break'] = 0
    data.loc[data['Close'] > data['Upper'], 'BOLL_Break'] = 1
    data.loc[data['Close'] < data['Lower'], 'BOLL_Break'] = -1
    return data.dropna()

def volume_breakout_signal(data, window=10, threshold=2):
    data = data.copy()
    data['VOL_Mean'] = data['Volume'].rolling(window).mean()
    # 确保索引对齐
    volume_aligned, vol_mean_aligned = data['Volume'].align(data['VOL_Mean'], axis=0)
    data['VOL_Break'] = (volume_aligned > vol_mean_aligned * threshold).astype(int)
    return data.dropna()

# 可继续扩展更多信号，如题材/行业动量（留接口）、涨停、主力流入等