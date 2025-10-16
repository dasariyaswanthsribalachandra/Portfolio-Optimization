from __future__ import annotations
import pandas as pd
import yfinance as yf

def download_prices(tickers, start: str = "2018-01-01", end: str | None = None) -> pd.DataFrame:
    # Robust yfinance fetch with auto_adjust
    data = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    prices = prices.dropna(how='all')
    # drop columns that are all NaN or too sparse
    prices = prices.dropna(axis=1, how='any')
    return prices

def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().dropna()

def get_data(tickers, start: str, end: str | None):
    prices = download_prices(tickers, start, end)
    returns = compute_returns(prices)
    return prices, returns
