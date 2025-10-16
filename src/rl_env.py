from __future__ import annotations
import numpy as np
import pandas as pd

class SimplePortfolioEnv:
    def __init__(self, returns: pd.DataFrame, window: int = 20, action_granularity: int = 5, cash_weight: float = 0.05):
        self.returns = returns.values
        self.tickers = list(returns.columns)
        self.window = window
        self.n_assets = returns.shape[1]
        self.t = window
        self.done = False
        self.action_space = self._build_action_space(action_granularity, cash_weight)
        self.state_dim = self.n_assets * window
        self.current_weights = np.ones(self.n_assets) / max(self.n_assets,1)

    def _build_action_space(self, granularity: int, cash_weight: float):
        actions = []
        if self.n_assets == 0:
            return np.array([[]])
        for i in range(self.n_assets):
            w = np.ones(self.n_assets) * ((1 - cash_weight) / max(self.n_assets - 1,1))
            w[i] = (1 - cash_weight)
            if self.n_assets > 1:
                w[(i+1) % self.n_assets] = ((1 - cash_weight) / max(self.n_assets - 1,1))
            actions.append(w)
        actions.append(np.ones(self.n_assets) / self.n_assets)  # equal weight
        return np.array(actions)

    def reset(self):
        self.t = self.window
        self.done = False
        self.current_weights = np.ones(self.n_assets) / max(self.n_assets,1)
        return self._obs()

    def _obs(self):
        if self.n_assets == 0 or self.t < self.window:
            return np.zeros((self.window * max(self.n_assets,1),), dtype=np.float32)
        window_returns = self.returns[self.t - self.window:self.t, :]
        return window_returns.flatten().astype(np.float32)

    def step(self, action_idx: int):
        if self.n_assets == 0 or self.t >= len(self.returns):
            self.done = True
            return None, 0.0, True, {}
        w = self.action_space[action_idx]
        r = self.returns[self.t, :]
        daily_port_ret = float((w * r).sum())
        self.current_weights = w
        self.t += 1
        if self.t >= len(self.returns):
            self.done = True
        reward = daily_port_ret
        return (self._obs() if not self.done else None), reward, self.done, {}

    @property
    def n_actions(self):
        return len(self.action_space)
