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


# set dataframe with options data iff options non-empty
options_df = fetch_options_data(tickers) 
if options_df is None or options_df.empty:
    formatted_df = None
else:
    clean_options_df = get_clean_options(options_df, current_prices)

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



# vol surface iff options data non-empty
st.subheader('Volatility Surface')

if formatted_df is not None:
    surface_fig, market_data = create_vol_surface_from_real_data(filtered_df, selected_ticker, current_prices)
    display_vol_surface_metrics(market_data, current_prices.get(selected_ticker, 100))
    if surface_fig is not None:
        st.plotly_chart(surface_fig, use_container_width=True)
    
else:
    st.info("Unable to create volatility surface. Try selecting a ticker with more options data.")


import streamlit as st
import time
from datetime import datetime
import uuid

# Simple test to see if cache persists across users/sessions

@st.cache_data(ttl=300)  # 5 minute cache
def test_cache_persistence():
    """This function should only run once per 5 minutes if cache is shared."""
    unique_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # Simulate expensive operation
    time.sleep(2)
    
    return {
        "unique_id": unique_id,
        "created_at": timestamp,
        "message": "This data was fetched fresh from 'API'"
    }

# Test page
st.title("🧪 Cache Persistence Test")

st.write("""
**Test Instructions:**
1. Load this page and note the unique_id and timestamp
2. Open a new incognito/private browser window  
3. Navigate to the same URL
4. Check if the unique_id and timestamp are the same

**If cache is shared:** Same ID and timestamp in both windows
**If cache is NOT shared:** Different ID and timestamp
""")

# Get cached data
with st.spinner("Loading... (2 second delay if cache miss)"):
    cached_data = test_cache_persistence()

st.success("✅ Data loaded!")

# Display results
col1, col2 = st.columns(2)
with col1:
    st.metric("Unique ID", cached_data["unique_id"])
with col2:
    st.metric("Created At", cached_data["created_at"])

st.info(f"Message: {cached_data['message']}")

# Instructions
st.divider()
st.subheader("🔍 How to Interpret Results")

st.write("""
**Scenario A - Cache IS Shared:**
- First user: Gets new ID (e.g., `abc123ef`) after 2-second delay
- Second user (different browser): Gets SAME ID (`abc123ef`) instantly
- ✅ Cache works across users!

**Scenario B - Cache NOT Shared:**  
- First user: Gets new ID (e.g., `abc123ef`) after 2-second delay
- Second user (different browser): Gets DIFFERENT ID (`xyz789gh`) after 2-second delay
- ❌ Each user has separate cache

**What this means for your options app:**
- Scenario A: Your dual caching strategy works great across users
- Scenario B: Every user will hit yfinance APIs separately
""")

# Additional info
with st.expander("🔧 Technical Details"):
    st.write(f"""
    - **Cache Function:** `test_cache_persistence()`
    - **TTL:** 5 minutes
    - **Current Time:** {datetime.now().strftime("%H:%M:%S")}
    - **Cache Key:** Based on function name and parameters (none in this case)
    """)

# Reset button for testing
if st.button("🗑️ Clear Cache (for testing)"):
    test_cache_persistence.clear()
    st.rerun()