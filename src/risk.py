from __future__ import annotations
import numpy as np
import pandas as pd
from .utils import compute_drawdown

def value_at_risk(returns: pd.Series, level: float = 0.95):
    return -np.percentile(returns.dropna(), (1 - level) * 100.0)

def conditional_var(returns: pd.Series, level: float = 0.95):
    var = value_at_risk(returns, level)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return -tail.mean()

def portfolio_returns(weights: dict[str, float], returns: pd.DataFrame) -> pd.Series:
    w = pd.Series(weights).reindex(returns.columns).fillna(0.0)
    port = (returns * w).sum(axis=1)
    return port

def risk_summary(weights: dict[str, float], returns: pd.DataFrame, rf: float = 0.0):
    port = portfolio_returns(weights, returns)
    if len(port) == 0:
        return {"annual_return": 0.0, "annual_volatility": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "VaR_95": 0.0, "CVaR_95": 0.0}, port, pd.Series(dtype=float)
    ann_ret = (1 + port).prod() ** (252 / max(len(port),1)) - 1
    ann_vol = port.std(ddof=0) * np.sqrt(252)
    sharpe = (ann_ret - rf) / (ann_vol + 1e-12)
    dd, max_dd = compute_drawdown(port)
    var95 = value_at_risk(port, 0.95)
    cvar95 = conditional_var(port, 0.95)
    return {
        "annual_return": float(ann_ret),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "VaR_95": float(var95),
        "CVaR_95": float(cvar95),
    }, port, dd
