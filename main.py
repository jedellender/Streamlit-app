import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm
import plotly.graph_objects as go

# import functions 
from data_fetcher import fetch_options_data, get_clean_options, fetch_data
from calculations import compute_greeks, display_price_chart, display_price_metrics
from iv_plotting import create_vol_surface, display_vol_surface_metrics, plot_vol_2d
from iv_solver import iv_calc
st.set_page_config(page_title="Compact Multi-Ticker Dashboard", layout="wide")

# init sidebar
st.sidebar.title("Inputs")
# Inputs 




# create sidebar for fallback data
with st.sidebar:
    st.subheader("Inputs")
    
    tickers_input = st.text_input(
        "Tickers (comma separated)",
        value=st.session_state.get('ticker_default', "TSLA, IONQ, SPY, AMD")
    ).upper()
    tickers = [t.strip() for t in tickers_input.split(",") if t.strip()][:5]
    
    benchmark_symbol = st.text_input("Benchmark (optional)", "SPY").upper()
    
    st.subheader("Data Source*")
    on = st.toggle("Toggle backup data**")
    
    # Update defaults when toggle changes
    if 'prev_toggle' not in st.session_state:
        st.session_state.prev_toggle = on
    
    if st.session_state.prev_toggle != on:
        st.session_state.ticker_default = "TSLA, IONQ, SPY, AMD"
        st.session_state.prev_toggle = on
        st.rerun()

     # Load data based on toggle state
    if on:
        options_df = pd.DataFrame()
        backup_liquid_options = [ "Backup_liquid_options/amd_2025-09-10T19-10_export.csv",
        "Backup_liquid_options/ionq_2025-09-10T19-10_export.csv",
        "Backup_liquid_options/spy_2025-09-10T19-10_export.csv",
        "Backup_liquid_options/tsla_2025-09-10T19-10_export.csv" ]
        
        for backup_export in backup_liquid_options:

            options_df = pd.concat([options_df, pd.read_csv(backup_export,
                parse_dates=["expirationDate"] )]
        )


    else:
        options_df = fetch_options_data(tickers)
    st.markdown("###### (*Live data may be unavailable outside of US market hours)")
    st.markdown("###### (**Example data/options chain as of 10/09/2025)")
    


# fetch equity data.
price_df, factor_df, current_prices = fetch_data(tickers)
returns = price_df.pct_change().dropna()


# Compute Beta ( default is spy)
if benchmark_symbol:
    try:
        bench_returns = yf.Ticker(benchmark_symbol).history(period="1y")["Close"].pct_change().dropna()
        bench_returns = bench_returns.reindex(returns.index).bfill()
        betas = {t: round(np.cov(returns[t], bench_returns)[0,1]/np.var(bench_returns), 2) for t in returns.columns}
        factor_df["Beta"] = pd.Series(betas)
    except:
        factor_df["Beta"] = 'N/A'
else:
    factor_df["Beta"] = 'N/A'  


# Factor Dashboard and correlation matrix 
col1, col2 = st.columns([3,1])
with col1: 
    st.subheader("Factors")
    st.dataframe(
        factor_df.style.format({
            "MarketCap (B)": "${:.2f}",
            "Forward P/E": "{:.2f}x",
            "Beta": "{:.2f}",
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



# choose equity for options data
col1, col2 = st.columns([1,1])
with col2:
    selected_ticker = st.selectbox("Select ticker:", tickers )#, label_visibility="collapsed")
with col1:
    st.subheader(f"{selected_ticker} options chain")


# display daily metrics 
display_price_metrics(price_df, selected_ticker)
t_max = 9 # st.slider('Days to Expiry range: ', 0, 772)
v_max = 2 # st.slider('Max vol', 0, 300)

filtered_df = None # init filtered df.

# process options data post selection of live / example data
if options_df is not None and not options_df.empty:
    clean_options_df = get_clean_options(options_df, current_prices, t_max, v_max, on=on)

    # clean IV with py_vollib
    clean_options_df = iv_calc(clean_options_df)

    # compute greeks
    greeks_df = compute_greeks(clean_options_df) 

    # filter for selected ticker and sort by volume
    filtered_df = greeks_df[ greeks_df['ticker'] == selected_ticker ].sort_values('volume', ascending=False)
    
    # ensure cols are numeric
    numeric_cols = [
        "strike", "volume", "impliedVolatility", "percentChange",
        "lastPrice", "change", "delta", "gamma", "theta", "vega",
        "rho", "moneyness", "intrinsic_value"
    ]
    filtered_df[numeric_cols] = filtered_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # create formatted display df
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
        "intrinsic_value":"{:.2f}",
        "bid": "${:.2f}",
        "ask": "${:.2f}",
        "bid_ask_spread": "${:.2f}",
        "bid_ask_spread_pct": "{:.2f}%"
    })

# make copy of filtered options dataframe for calculations
    calc_df = filtered_df.copy()

# toggle display options data
with st.expander("Toggle options data"):
        if calc_df is not None:
            st.dataframe(
                calc_df, 
                use_container_width=True,
            )
        else:
            st.info("No options data available for the selected ticker.")

# vol surface iff options data non-empty
st.subheader(f" {selected_ticker} Volatility Surface")
if calc_df is not None:
    surface_fig, market_data = create_vol_surface(filtered_df, selected_ticker, current_prices)
    if surface_fig is not None:
        st.plotly_chart(surface_fig, use_container_width=True)
        
    display_vol_surface_metrics(market_data, current_prices)
    
else:
    st.info("Unable to create volatility surface. Try selecting a ticker with more options data.")

if filtered_df is None or filtered_df.empty:
        st.warning("No data available for volatility smile.")
else:
    smile_fig = plot_vol_2d(filtered_df, selected_ticker, current_prices)
    st.plotly_chart(smile_fig)
