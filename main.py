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
st.subheader('Historical equities ')
on = st.toggle("Toggle chart", key='price_chart', value=True)
if on:
    display_price_chart(price_df)


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
from datetime import datetime
from db import get_connection, init_db, save_options, load_options
import yfinance as yf

def test_database_page():
    """Add this to your main app.py or run as standalone test"""
    
    st.title("🔧 Database Test Page")
    
    # Test 1: Connection
    st.header("1️⃣ Test Connection")
    if st.button("Test Database Connection"):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            st.success(f"✅ Connected! PostgreSQL version: {version}")
            cursor.close()
            conn.close()
        except Exception as e:
            st.error(f"❌ Connection failed: {e}")
    
    # Test 2: Check Table
    st.header("2️⃣ Check Table")
    if st.button("Check if table exists"):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'options_cache'
                )
            """)
            exists = cursor.fetchone()[0]
            if exists:
                st.success("✅ Table 'options_cache' exists!")
                
                # Count records
                cursor.execute("SELECT COUNT(*) FROM options_cache")
                count = cursor.fetchone()[0]
                st.info(f"📊 Total records in database: {count}")
                
                # Show unique tickers
                cursor.execute("SELECT DISTINCT ticker FROM options_cache")
                tickers = cursor.fetchall()
                if tickers:
                    st.write("Tickers in database:", [t[0] for t in tickers])
            else:
                st.warning("Table doesn't exist, creating...")
                init_db()
                st.success("✅ Table created!")
            
            cursor.close()
            conn.close()
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Test 3: Save Test Data
    st.header("3️⃣ Test Save Data")
    test_ticker = st.text_input("Enter a ticker to test save:", "AAPL")
    
    if st.button("Fetch and Save Options"):
        with st.spinner(f"Fetching {test_ticker} options..."):
            try:
                # Fetch real data
                ticker = yf.Ticker(test_ticker)
                if ticker.options:
                    exp_date = ticker.options[0]
                    opt_chain = ticker.option_chain(exp_date)
                    
                    # Get first 5 calls as test
                    test_df = opt_chain.calls.head(5).copy()
                    test_df['ticker'] = test_ticker
                    test_df['optionType'] = 'Call'
                    test_df['expirationDate'] = exp_date
                    
                    # Show what we're saving
                    st.write("Sample data to save:")
                    st.dataframe(test_df[['strike', 'lastPrice', 'volume']].head())
                    
                    # Save it
                    save_options(test_df)
                    st.success(f"✅ Saved {len(test_df)} options for {test_ticker}")
                else:
                    st.warning(f"No options found for {test_ticker}")
            except Exception as e:
                st.error(f"❌ Error: {e}")
    
    # Test 4: Load Data
    st.header("4️⃣ Test Load Data")
    if st.button("Load All Data from Database"):
        try:
            # Get all tickers in database
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT ticker FROM options_cache")
            tickers = [t[0] for t in cursor.fetchall()]
            cursor.close()
            conn.close()
            
            if tickers:
                st.write(f"Loading data for: {tickers}")
                df = load_options(tickers)
                
                if not df.empty:
                    st.success(f"✅ Loaded {len(df)} records")
                    
                    # Show summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Unique Tickers", df['ticker'].nunique())
                    with col3:
                        st.metric("Expiration Dates", df['expirationDate'].nunique())
                    
                    # Show sample data
                    st.write("Sample loaded data:")
                    st.dataframe(df[['ticker', 'strike', 'lastPrice', 'volume', 'stored_at']].head(10))
                else:
                    st.warning("Database returned no data")
            else:
                st.info("No data in database yet. Try saving some first!")
                
        except Exception as e:
            st.error(f"❌ Error loading: {e}")
    
    # Test 5: Live Status
    st.header("5️⃣ Database Status")
    if st.button("Get Database Stats"):
        try:
            conn = get_connection()
            cursor = conn.cursor()
            
            # Get stats
            cursor.execute("""
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    MIN(stored_at) as oldest_record,
                    MAX(stored_at) as newest_record
                FROM options_cache
            """)
            stats = cursor.fetchone()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", stats[0])
                st.metric("Unique Tickers", stats[1])
            with col2:
                if stats[2]:
                    st.metric("Oldest Record", stats[2].strftime("%Y-%m-%d %H:%M"))
                    st.metric("Newest Record", stats[3].strftime("%Y-%m-%d %H:%M"))
            
            # Database size (Supabase specific)
            cursor.execute("SELECT pg_database_size(current_database())/1024/1024 as size_mb")
            size = cursor.fetchone()[0]
            st.info(f"📦 Database size: {size:.2f} MB / 500 MB (free limit)")
            
            cursor.close()
            conn.close()
        except Exception as e:
            st.error(f"❌ Error: {e}")
    
    # Quick cleanup option
    st.header("🧹 Cleanup")
    if st.button("Clear All Data", type="secondary"):
        if st.checkbox("I'm sure - delete everything"):
            try:
                conn = get_connection()
                cursor = conn.cursor()
                cursor.execute("TRUNCATE TABLE options_cache")
                conn.commit()
                st.success("✅ All data cleared")
                cursor.close()
                conn.close()
            except Exception as e:
                st.error(f"❌ Error: {e}")

# If running this as part of your main app, add to sidebar:
# In your main app.py, add:
# if st.sidebar.button("Database Test"):
#     test_database_page()

# Or run standalone:
if __name__ == "__main__":
    test_database_page()

with st.sidebar.expander("🔧 Database Testing"):
    if st.button("Open Test Page"):
        test_database_page()  # The function from above
        