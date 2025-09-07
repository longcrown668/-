import matplotlib.pyplot as plt

def plot_signal(data, signal_col, title="信号点"):
    plt.figure(figsize=(12,6))
    plt.plot(data['Close'], label='Close')
    sig_idx = data[data[signal_col]==1].index
    plt.scatter(sig_idx, data.loc[sig_idx, 'Close'], color='r', marker='^', label='Signal')
    plt.legend()
    plt.title(title)
    plt.show()

def plot_pnl_curve(pnl, title="策略收益曲线"):
    plt.figure(figsize=(12,6))
    plt.plot(pnl, label='PnL')
    plt.title(title)
    plt.legend()
    plt.show()