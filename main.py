import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import nltk
from scipy.stats import norm
import seaborn

# import functions 
from data_fetcher import fetch_options_data, get_clean_options, fetch_data
from calculations import compute_greeks, display_price_chart, display_price_metrics, display_all_price_metrics
from iv_plotting import plot_3d_copula

st.set_page_config(page_title="Compact Multi-Ticker Dashboard", layout="wide")
st.sidebar.title("Inputs")

# Inputs 
tickers_input = st.sidebar.text_input("Enter up to 5 tickers (comma separated)", "QBTS, IONQ, RGTI, QUBT ").upper()
benchmark_symbol = st.sidebar.text_input("Benchmark (optional)", 'SPY').upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()][:5]
option_expiry = '2025-09-26' #date.today() + timedelta(days=30)


# fetch equity data.
price_df, factor_df, current_prices = fetch_data(tickers)
returns = price_df.pct_change().dropna()

# set dataframe with options data
options_df = fetch_options_data(tickers) 
clean_options_df = get_clean_options(options_df, current_prices)

# Compute Beta ( default is spy)
if benchmark_symbol:
    try:
        bench_returns = yf.Ticker(benchmark_symbol).history(period="1y")["Close"].pct_change().dropna()
        bench_returns = bench_returns.reindex(returns.index).fillna(method="ffill")
        betas = {t: np.cov(returns[t], bench_returns)[0,1]/np.var(bench_returns) for t in returns.columns}
        factor_df["Beta"] = pd.Series(betas)
    except:
        factor_df["Beta"] = np.nan
else:
    factor_df["Beta"] = np.nan


# Factor Dashboard and correlation matrix 
col1, col2 = st.columns([3,1])
with col1: 
    st.subheader("Factors")
    st.dataframe(
        factor_df.style.format({
            "MarketCap (B)": "${:.2f}",
            "Forward P/E": "{:.2f}x",
            "Beta to bench": "{:.2f}",
            "styled_corr": "{:.2f}"
        })
    )
with col2: 
    st.subheader("Correlation Matrix")
    corr = returns.corr().style.background_gradient(cmap='RdYlBu', axis=None, vmax=1, vmin=-1).format("{:.2f}")
    st.dataframe(corr,width=400)

# display chart
on = st.toggle("Toggle chart")
if on:
    display_price_chart(price_df, 'Equities historical price')


# choose equity for options datra
col1, col2 = st.columns([1,1])
with col2:
    selected_ticker = st.selectbox("Select ticker:", tickers )#, label_visibility="collapsed")
with col1:
    st.subheader(f"{selected_ticker} Data and options options chain")

#display_all_price_metrics(price_df)

# display daily metrics 

# col1, col2 = st.columns([1,1])
# with col2:
#prices_df, infos_df, current_prices = fetch_data(['AAPL', 'MSFT'])
display_price_metrics(price_df, selected_ticker)

# format and display options/greeks 
greeks_df = compute_greeks(clean_options_df) 
filtered_df = greeks_df[ greeks_df['ticker'] == selected_ticker ].sort_values('volume', ascending=False)

formatted_df = filtered_df.style.format({
    "expirationDate": lambda x: x.strftime('%d/%m/%y') if pd.notna(x) else "",
    "strike": "${:.2f}",
    "volume":"{:.2f}",
    "impliedVolatility": "{:.1%}",    
    "percentChange": "{:.2f}%",
    "lastPrice": "${:.3f}",
    "change": "${:.2f}", 
    "delta": "{:.3f}",
    "gamma": "{:.4f}",
    "theta": "{:.4f}",
    "vega": "{:.4f}",
    "rho": "{:.4f}",
    "moneyness": "{:.3f}",
    "intrinsic_value":"{:.2f}"
})
st.dataframe(formatted_df)



# vol surface

st.subheader('Volatility Surface')


fig_3d = plot_3d_copula(filtered_df, 'impliedVolatility', 'Call')
if fig_3d:
    st.pyplot(fig_3d)
else: 
    st.error('insufficient data')