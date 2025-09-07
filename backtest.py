import pandas as pd
import numpy as np

def signal_backtest(data, signal_col, horizon=5, min_trigger=5):
    stats = []
    for i in range(len(data) - horizon):
        if data.iloc[i][signal_col] == 1:
            future = data.iloc[i+1:i+1+horizon]['Close']
            if future.empty:
                continue
            future_return = (future.iloc[-1] / data.iloc[i]['Close'] - 1) * 100
            stats.append(future_return)
    triggers = len(stats)
    if triggers >= min_trigger:
        win_rate = sum([s > 0 for s in stats]) / triggers * 100
        avg_return = np.mean(stats)
        max_drawdown = np.min(stats)
        return {
            '信号': signal_col,
            '触发次数': triggers,
            '胜率%': round(win_rate, 2),
            '平均收益%': round(avg_return, 2),
            '最大回撤%': round(max_drawdown, 2)
        }
    else:
        return None

def multi_signal_backtest(data, signal_cols, horizon=5, min_trigger=5):
    results = []
    for col in signal_cols:
        res = signal_backtest(data, col, horizon, min_trigger)
        if res:
            results.append(res)
    return pd.DataFrame(results)