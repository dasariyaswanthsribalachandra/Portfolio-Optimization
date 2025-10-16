from __future__ import annotations
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

def optimize_mpt(prices: pd.DataFrame, objective: str = "max_sharpe", risk_free_rate: float = 0.0):
    mu = expected_returns.mean_historical_return(prices)  # annualized
    S = risk_models.sample_cov(prices)                    # annualized covariance
    ef = EfficientFrontier(mu, S)
    if objective == "max_sharpe":
        ef.max_sharpe(risk_free_rate=risk_free_rate)
    elif objective == "min_volatility":
        ef.min_volatility()
    else:
        raise ValueError("objective must be 'max_sharpe' or 'min_volatility'")
    weights = ef.clean_weights()
    perf = ef.portfolio_performance(verbose=False, risk_free_rate=risk_free_rate)
    return weights, perf

def efficient_frontier_points(prices: pd.DataFrame, num_points: int = 40):
    # Sample along returns range for stability
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    min_ret, max_ret = float(mu.min()), float(mu.max())
    targets = np.linspace(min_ret, max_ret, num_points)
    risks, rets = [], []
    ef = EfficientFrontier(mu, S)
    for tr in targets:
        try:
            ef.efficient_return(target_return=tr)
            ret, risk, _ = ef.portfolio_performance(verbose=False)
            risks.append(risk)
            rets.append(ret)
        except Exception:
            # skip infeasible points
            continue
    return pd.DataFrame({'risk': risks, 'return': rets})

def discrete_allocation(prices: pd.DataFrame, weights: dict, total_portfolio_value: float = 10000):
    latest = get_latest_prices(prices)
    da = DiscreteAllocation(weights, latest, total_portfolio_value=total_portfolio_value)
    allocation, leftover = da.greedy_portfolio()
    return allocation, leftover
