import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from typing import List, Dict

# 可选：深度学习
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False

# ------- 动态信号回测 -------
def dynamic_signal_backtest(
    ticker: str,
    lookback: int = 90,
    horizon: int = 5,
    min_trigger: int = 5,
    calculate_indicators=None,
    check_strategies=None,
) -> pd.DataFrame:
    data = yf.download(ticker, period=f"{lookback+60}d", interval="1d", auto_adjust=True, progress=False)
    if data is None or len(data) < lookback or calculate_indicators is None or check_strategies is None:
        return pd.DataFrame()
    data = calculate_indicators(data)
    results: Dict[str, Dict] = {}
    for i in range(len(data) - horizon):
        today = data.iloc[i]
        future = data.iloc[i+1:i+1+horizon]["Close"]
        if future.empty:
            continue
        signals = check_strategies(today, today["Close"])
        future_return = (future.iloc[-1] / today["Close"] - 1) * 100
        for sig in signals:
            if sig not in results:
                results[sig] = {"triggers": 0, "wins": 0, "returns": []}
            results[sig]["triggers"] += 1
            if future_return > 0:
                results[sig]["wins"] += 1
            results[sig]["returns"].append(future_return)
    stats = []
    for sig, val in results.items():
        if val["triggers"] >= min_trigger:
            win_rate = val["wins"] / val["triggers"] * 100
            avg_return = np.mean(val["returns"])
            median_return = np.median(val["returns"])
            stats.append({
                "信号": sig,
                "胜率%": round(win_rate, 2),
                "平均收益%": round(avg_return, 2),
                "中位数收益%": round(median_return, 2),
                "触发次数": val["triggers"],
                "权重": round(win_rate * np.log1p(val["triggers"]), 2)
            })
    df = pd.DataFrame(stats).sort_values(by=["权重", "胜率%"], ascending=False)
    return df

# ------- 特征工程 -------
def extract_ml_features(data: pd.DataFrame, feature_cols: List[str], horizon: int = 5):
    X, y = [], []
    for i in range(len(data) - horizon):
        feat = data.iloc[i][feature_cols].values
        target = 1 if data.iloc[i + horizon]["Close"] > data.iloc[i]["Close"] else 0
        X.append(feat)
        y.append(target)
    return np.array(X), np.array(y)

# ------- 集成学习：Voting/Stacking -------
def train_ensemble(X, y):
    rf = RandomForestClassifier(n_estimators=100)
    lr = LogisticRegression(max_iter=300)
    # Voting
    voting = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
    voting.fit(X, y)
    # Stacking
    stack = StackingClassifier(estimators=[('rf', rf), ('lr', lr)], final_estimator=LogisticRegression())
    stack.fit(X, y)
    return {"voting": voting, "stack": stack}

# ------- 深度学习：MLP -------
def train_mlp(X, y, input_dim, epochs=20, batch_size=16):
    if not HAS_TF:
        print("TensorFlow/Keras not installed!")
        return None
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_dim=input_dim),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

# ------- 综合主流程 -------
def adaptive_ai_selector(
    stock_list: List[str],
    calculate_indicators,
    check_strategies,
    feature_cols: List[str],
    lookback: int = 120,
    horizon: int = 5,
    winrate_threshold: float = 55,
    top_n: int = 5,
    use_mlp: bool = True,
    use_ensemble: bool = True
) -> pd.DataFrame:
    results = []
    for ticker in stock_list:
        try:
            data = yf.download(ticker, period=f"{lookback+60}d", interval="1d", auto_adjust=True, progress=False)
            if data is None or len(data) < lookback:
                continue
            data = calculate_indicators(data)
            today = data.iloc[-1]
            price = float(today["Close"])
            # 动态信号回测
            signal_stats = dynamic_signal_backtest(
                ticker, lookback=lookback, horizon=horizon,
                calculate_indicators=calculate_indicators, check_strategies=check_strategies
            )
            if signal_stats.empty:
                continue
            best_signals = signal_stats[signal_stats["胜率%"] >= winrate_threshold].head(3)
            rec_signals = []
            for _, row in best_signals.iterrows():
                sig = row["信号"]
                if sig in check_strategies(today, price):
                    rec_signals.append(sig)
            # 特征工程
            X, y = extract_ml_features(data, feature_cols, horizon)
            if len(X) < 10:
                continue
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            # 集成
            if use_ensemble:
                models = train_ensemble(X_train, y_train)
                voting_pred = models["voting"].predict_proba([today[feature_cols].values])[0][1]
            else:
                voting_pred = None
            # 深度学习
            if use_mlp and HAS_TF:
                mlp = train_mlp(X_train, y_train, input_dim=len(feature_cols), epochs=20)
                mlp_pred = float(mlp.predict([today[feature_cols].values])[0][0])
            else:
                mlp_pred = None
            # 输出
            results.append({
                "股票": ticker,
                "价格": price,
                "推荐信号": "、".join(rec_signals),
                "近期胜率%": best_signals["胜率%"].values[0] if not best_signals.empty else None,
                "触发次数": best_signals["触发次数"].values[0] if not best_signals.empty else None,
                "集成AI上涨概率": round(voting_pred*100,2) if voting_pred is not None else None,
                "深度学习上涨概率": round(mlp_pred*100,2) if mlp_pred is not None else None,
                "建议": f"信号: {rec_signals}；AI预测上涨{round(mlp_pred*100,2) if mlp_pred is not None else 'N/A'}%"
            })
        except Exception as e:
            print(f"{ticker} failed: {e}")
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(by=["集成AI上涨概率", "深度学习上涨概率", "近期胜率%"], ascending=False).head(top_n)
    return df
