import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date

from pypfopt import EfficientFrontier, risk_models, expected_returns

st.set_page_config(page_title="Portfolio Optimizer (MPT)", layout="wide")
st.title("Portfolio Optimization & Risk Management (MPT Only)")
st.write("App started successfully!")

# Sidebar inputs
with st.sidebar:
    st.header("Settings")
    tickers = st.text_input("Tickers (comma-separated)", "AAPL,MSFT,GOOGL").split(",")
    tickers = [t.strip().upper() for t in tickers if t.strip()]
    start = st.date_input("Start Date", date(2019, 1, 1))
    end = st.date_input("End Date", date.today())
    use_demo = st.checkbox("Use demo data", value=True)
    objective = st.selectbox("MPT Objective", ["max_sharpe", "min_volatility"])
    rf = st.number_input("Risk-free rate (annual)", value=0.02, step=0.01, format="%.2f")

# Load data
if use_demo:
    dates = pd.date_range(start="2020-01-01", periods=250, freq="B")
    prices = pd.DataFrame({
        "AAPL": np.linspace(100, 200, len(dates)) + np.random.randn(len(dates))*5,
        "MSFT": np.linspace(150, 250, len(dates)) + np.random.randn(len(dates))*5,
        "GOOGL": np.linspace(1200, 1800, len(dates)) + np.random.randn(len(dates))*20,
    }, index=dates)
else:
    import yfinance as yf
    data = yf.download(tickers, start=start, end=end, progress=False)["Adj Close"]
    prices = data.dropna()

st.subheader("Price Chart")
st.line_chart(prices)

# Optimize with MPT
returns = prices.pct_change().dropna()
mu = expected_returns.mean_historical_return(prices)
S = risk_models.sample_cov(prices)
ef = EfficientFrontier(mu, S)

if objective == "max_sharpe":
    ef.max_sharpe(risk_free_rate=rf)
else:
    ef.min_volatility()

weights = ef.clean_weights()
perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)

# Show results
st.subheader("Optimized Weights")
w_df = pd.DataFrame.from_dict(weights, orient="index", columns=["Weight"]).sort_values("Weight", ascending=False)
st.dataframe(w_df.style.format({"Weight": "{:.2%}"}))

st.subheader("Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Annual Return", f"{perf[0]:.2%}")
col2.metric("Annual Volatility", f"{perf[1]:.2%}")
col3.metric("Sharpe Ratio", f"{perf[2]:.2f}")

# Efficient frontier (sampled points)
st.subheader("Efficient Frontier")
returns_list, risks = [], []
for r in np.linspace(mu.min(), mu.max(), 25):
    try:
        ef = EfficientFrontier(mu, S)
        ef.efficient_return(r)
        perf = ef.portfolio_performance(verbose=False, risk_free_rate=rf)
        returns_list.append(perf[0])
        risks.append(perf[1])
    except:
        pass

ef_df = pd.DataFrame({"risk": risks, "return": returns_list})
fig = px.scatter(ef_df, x="risk", y="return", title="Efficient Frontier")
st.plotly_chart(fig, use_container_width=True)

st.caption("© Portfolio Optimizer (MPT Only) — Educational use only.")
