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


########### TEST
st.title("🔧 Database Functional Test")

# Make sure table exists
init_db()


import streamlit as st
from sqlalchemy import text  # Add this import at the top with other imports
from db import get_connection, save_options, init_db, test_connection

# Database Testing Section
st.sidebar.divider()
st.sidebar.subheader("🔧 Database Tools")

# Quick connection test
if st.sidebar.button("Test DB Connection"):
    with st.spinner("Testing database..."):
        if test_connection():
            st.sidebar.success("✅ Database working")
        else:
            st.sidebar.error("❌ Database issue")

# Show cached data info
if st.sidebar.button("Show Cached Data"):
    try:
        engine = get_connection()
        with engine.connect() as conn:
            # Get summary
            result = conn.execute(text("""
                SELECT 
                    ticker,
                    COUNT(*) as options_count,
                    MIN(expiration_date) as nearest_expiry,
                    MAX(expiration_date) as furthest_expiry,
                    MAX(stored_at) as last_updated
                FROM options_cache
                GROUP BY ticker
                ORDER BY ticker
            """))
            
            summary = pd.DataFrame(result.fetchall(), 
                                 columns=['Ticker', 'Options Count', 'Nearest Expiry', 
                                         'Furthest Expiry', 'Last Updated'])
            
            if not summary.empty:
                st.sidebar.dataframe(summary)
            else:
                st.sidebar.info("No cached data")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Force cache refresh
if st.sidebar.button("🔄 Force Refresh Cache"):
    # Clear cache
    st.cache_data.clear()
    st.rerun()

# Manual database operations (only show in debug mode)
debug_mode = st.sidebar.checkbox("Debug Mode", value=False)

if debug_mode:
    st.sidebar.divider()
    st.sidebar.write("### Debug Operations")
    
    # Clear specific ticker
    ticker_to_clear = st.sidebar.text_input("Clear ticker from DB:")
    if st.sidebar.button("Clear Ticker") and ticker_to_clear:
        try:
            engine = get_connection()
            with engine.begin() as conn:
                result = conn.execute(
                    text("DELETE FROM options_cache WHERE ticker = :ticker"),
                    {"ticker": ticker_to_clear.upper()}
                )
                st.sidebar.success(f"Cleared {result.rowcount} rows for {ticker_to_clear}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")
    
    # Show recent errors
    if st.sidebar.button("Show DB Schema"):
        try:
            engine = get_connection()
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns
                    WHERE table_name = 'options_cache'
                    ORDER BY ordinal_position
                """))
                schema = pd.DataFrame(result.fetchall(), 
                                    columns=['Column', 'Type', 'Nullable'])
                st.sidebar.dataframe(schema)

        except Exception as e:
            st.sidebar.error(f"Error: {e}")