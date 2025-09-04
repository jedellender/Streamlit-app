import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm


# import functions 
from data_fetcher import fetch_options_data, get_clean_options, fetch_data
from calculations import compute_greeks, display_price_chart, display_price_metrics
from iv_plotting import create_vol_surface_from_real_data, display_vol_surface_metrics
from db import get_connection, save_options, load_options, init_db
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
save_options(options_df)  # Save fetched options to the database

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
with st.expander(label='Toggle chart', expanded=True):
    if not price_df.empty:
        display_price_chart(price_df)
    else:
        st.info("No price data available.")


# choose equity for options datra
col1, col2 = st.columns([1,1])
with col2:
    selected_ticker = st.selectbox("Select ticker:", tickers )#, label_visibility="collapsed")
with col1:
    st.subheader(f"{selected_ticker} options chain")


# display daily metrics 

display_price_metrics(price_df, selected_ticker)


# format options data
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


# toggle display options data
with st.expander("Toggle options data"):
        if formatted_df is not None:
            st.dataframe(
                formatted_df, 
                use_container_width=True,
            )
        else:
            st.info("No options data available for the selected ticker.")



# vol surface
st.subheader('Volatility Surface')

surface_fig, market_data = create_vol_surface_from_real_data(filtered_df, selected_ticker, current_prices)

if surface_fig is not None:
    display_vol_surface_metrics(market_data, current_prices.get(selected_ticker, 100))
    st.plotly_chart(surface_fig, use_container_width=True)
    
    # Optional: Show sample of the data used
    # with st.expander("📊 Data Used for Surface"):
    #     if market_data is not None:
    #         st.dataframe(
    #             market_data.head(10).round(4), 
    #             use_container_width=True
    #         )
else:
    st.info("Unable to create volatility surface. Try selecting a ticker with more options data.")


import streamlit as st
import pandas as pd
import yfinance as yf
from db import get_connection, save_options, load_options, init_db

st.title("🔧 Database Functional Test")

# Make sure table exists
init_db()

engine = get_connection()

# --- 1️⃣ Enter ticker ---
test_ticker = st.text_input("Enter a ticker to test save/load:", "AAPL")

# --- 2️⃣ Fetch & Save ---
if st.button("Fetch and Save Options"):
    with st.spinner(f"Fetching {test_ticker} options..."):
        try:
            ticker = yf.Ticker(test_ticker)
            if ticker.options:
                exp_date = ticker.options[0]
                opt_chain = ticker.option_chain(exp_date)

                # First 5 calls only
                df = opt_chain.calls.head(5).copy()
                df['ticker'] = test_ticker
                df['optionType'] = 'Call'
                df['expirationDate'] = exp_date

                st.write("Sample data to save:")
                st.dataframe(df[['strike', 'lastPrice', 'volume']])

                save_options(df)
                st.success(f"✅ Saved {len(df)} options for {test_ticker}")
            else:
                st.warning(f"No options found for {test_ticker}")
        except Exception as e:
            st.error(f"❌ Error fetching/saving: {e}")

# --- 3️⃣ Load & Verify ---
if st.button("Load Options from DB"):
    try:
        loaded_df = load_options([test_ticker])
        if not loaded_df.empty:
            st.success(f"✅ Loaded {len(loaded_df)} records for {test_ticker}")
            st.dataframe(loaded_df[['ticker','strike','lastPrice','volume','stored_at']])
        else:
            st.warning("No data found in DB yet. Try saving first!")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")

# --- 4️⃣ Cleanup (optional) ---
if st.button("Clear Test Data"):
    with st.spinner("Clearing test data..."):
        try:
            with engine.begin() as conn:
                conn.execute("DELETE FROM options_cache WHERE ticker = :t", {"t": test_ticker})
            st.success(f"✅ Cleared data for {test_ticker}")
        except Exception as e:
            st.error(f"❌ Error clearing data: {e}")

