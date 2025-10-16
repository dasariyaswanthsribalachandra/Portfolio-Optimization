import numpy as np
import pandas as pd

def annualize_returns(daily_returns: pd.Series | pd.DataFrame, trading_days: int = 252):
    return daily_returns.mean() * trading_days

def annualize_volatility(daily_returns: pd.Series | pd.DataFrame, trading_days: int = 252):
    return daily_returns.std(ddof=0) * np.sqrt(trading_days)

def sharpe_ratio(daily_returns: pd.Series | pd.DataFrame, rf: float = 0.0, trading_days: int = 252):
    ann_ret = annualize_returns(daily_returns, trading_days)
    ann_vol = annualize_volatility(daily_returns, trading_days)
    return (ann_ret - rf) / (ann_vol + 1e-12)

def compute_drawdown(series: pd.Series):
    cum = (1 + series).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    return dd, max_dd

def to_returns(prices: pd.DataFrame):
    return prices.pct_change().dropna()
