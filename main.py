import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

from data_fetcher import fetch_options_data, get_clean_options, fetch_equity_data
from calculations import compute_greeks, display_price_chart, display_price_metrics
from iv_plotting import create_vol_surface_from_real_data, display_vol_surface_metrics
from db import init_db, load_options_from_cache

# --- Page Config & Initialization ---
st.set_page_config(page_title="Options Dashboard", layout="wide")
init_db()

if 'options_data' not in st.session_state:
    st.session_state.options_data = pd.DataFrame()

# --- Sidebar Controls ---
st.sidebar.title("Inputs")
tickers_input = st.sidebar.text_input("Tickers (comma separated)", "AAPL,GOOG,NVDA,TSLA").upper()
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

st.sidebar.divider()
st.sidebar.subheader("Data Control")

if st.sidebar.button("Fetch Live Options Data", use_container_width=True, type="primary"):
    with st.spinner("Fetching live data and updating cache..."):
        st.session_state.options_data = fetch_options_data(tickers)
        if st.session_state.options_data.empty:
            st.warning("Failed to fetch live data. Try loading from cache.")

if st.sidebar.button("Load from DB Cache", use_container_width=True):
    with st.spinner("Loading from database..."):
        st.session_state.options_data = load_options_from_cache()
        if not st.session_state.options_data.empty:
            st.success("Loaded data from cache.")
        else:
            st.warning("Database cache is empty.")

# --- Main Page Display ---
st.title("Options & Volatility Dashboard")

if not tickers:
    st.info("Enter at least one ticker symbol to begin.")
    st.stop()

# --- Equity Data Section ---
price_df, factor_df, current_prices = fetch_equity_data(tickers)

if not price_df.empty:
    st.subheader("Factor Dashboard")
    st.dataframe(factor_df.style.format({
        "MarketCap (B)": "${:,.2f}", "Forward P/E": "{:.2f}x"
    }))
else:
    st.warning("Could not fetch equity data. Some features will be unavailable.")

# --- Options Analysis Section ---
st.divider()
st.subheader("Options Analysis")

if st.session_state.options_data.empty:
    st.info("No options data is loaded. Use the sidebar to fetch live data or load from the cache.")
else:
    options_df = st.session_state.options_data
    available_tickers = sorted(options_df['ticker'].unique())
    
    col1, col2 = st.columns([1, 3])
    selected_ticker = col1.selectbox("Select ticker for analysis:", available_tickers)

    if selected_ticker and selected_ticker in current_prices:
        display_price_metrics(price_df, selected_ticker)
        
        clean_df = get_clean_options(options_df, current_prices)
        greeks_df = compute_greeks(clean_df)
        
        filtered_df = greeks_df[greeks_df['ticker'] == selected_ticker]

        with st.expander(f"Options Chain for {selected_ticker}", expanded=True):
            st.dataframe(filtered_df.style.format(precision=2), use_container_width=True)

        st.subheader(f'Implied Volatility Surface for {selected_ticker}')
        surface_fig, _ = create_vol_surface_from_real_data(
            filtered_df, selected_ticker, current_prices.get(selected_ticker, 0)
        )
        if surface_fig:
            st.plotly_chart(surface_fig, use_container_width=True)
        else:
            st.info("Not enough data to create a volatility surface.")

