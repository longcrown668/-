import yfinance as yf
import pandas as pd
from signals import ma_signals, rsi_signal, macd_signal, bollinger_signal, volume_breakout_signal
from backtest import multi_signal_backtest
from ai_models import extract_features, train_voting_rf_lr, train_mlp
from visualize import plot_signal, plot_pnl_curve

def calculate_all_signals(data):
    data = ma_signals(data)
    data = rsi_signal(data)
    data = macd_signal(data)
    data = bollinger_signal(data)
    data = volume_breakout_signal(data)
    return data

def main(stock_list, horizon=5, min_trigger=5, winrate_thres=55, top_n=5):
    all_results = []
    for ticker in stock_list:
        data = yf.download(ticker, period="180d", auto_adjust=True, progress=False)
        data = calculate_all_signals(data)
        signal_cols = ['MA_Cross', 'RSI_signal', 'MACD_Cross', 'BOLL_Break', 'VOL_Break']
        stats = multi_signal_backtest(data, signal_cols, horizon, min_trigger)
        if stats.empty:
            continue
        best = stats[stats['胜率%'] >= winrate_thres].sort_values(by='胜率%', ascending=False).head(1)
        if not best.empty:
            feature_cols = ['MA_short', 'MA_long', 'RSI', 'MACD', 'Upper', 'Lower', 'VOL_Mean']
            X, y = extract_features(data, feature_cols, horizon)
            if len(X) > 10:
                mdl = train_voting_rf_lr(X, y)
                pred = mdl.predict_proba([data.iloc[-1][feature_cols].values])[0][1]
            else:
                pred = None
            all_results.append({
                '股票': ticker,
                '最佳信号': best['信号'].values[0],
                '胜率%': best['胜率%'].values[0],
                '平均收益%': best['平均收益%'].values[0],
                'AI上涨概率': round(pred*100,2) if pred is not None else None
            })
    df = pd.DataFrame(all_results).sort_values(by=['AI上涨概率','胜率%'], ascending=False).head(top_n)
    print(df)
    return df

if __name__ == "__main__":
    # 示例股票池
    stock_list = ['AAPL', 'MSFT', 'GOOG']
    main(stock_list)