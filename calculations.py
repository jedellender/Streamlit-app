import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import norm
import plotly.express as px

@st.cache_data(ttl=900)
def compute_greeks(clean_options_df, risk_free_rate=0.04): # function to compute greeks data
    
    df = clean_options_df.copy()
    
    # Black Scholes inputs 
    S = df['current_price'] 
    K = df['strike']
    T = df['time_to_expiry']
    r = risk_free_rate
    sig = df['impliedVolatility']
    # calc d1 and d2
    d1 = (np.log(S/K) + (r + (sig**2)/2)*T)/ (sig *np.sqrt(T))
    d2 = d1- sig*np.sqrt(T)

    # standard normal CDF and PDF
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    n_d1 = norm.pdf(d1) # pdf for gamma and vega


    # Initialize Greek columns
    df['delta'] = 0.0
    df['gamma'] = 0.0
    df['theta'] = 0.0
    df['vega'] = 0.0
    df['rho'] = 0.0

    call_mask = df['optionType'] == 'Call'

    # Call Greeks
    df.loc[call_mask, 'delta'] = N_d1[call_mask]
    df.loc[call_mask, 'theta'] = (-(S[call_mask] * n_d1[call_mask] * sig[call_mask]) / (2 * np.sqrt(T[call_mask])) 
                                  - r * K[call_mask] * np.exp(-r * T[call_mask]) * N_d2[call_mask]) / 252
    df.loc[call_mask, 'rho'] = K[call_mask] * T[call_mask] * np.exp(-r * T[call_mask]) * N_d2[call_mask] / 100
    
# Gamma and Vega are the same for calls and puts
    df['gamma'] = n_d1 / (S * sig * np.sqrt(T))
    df['vega'] = S * n_d1 * np.sqrt(T) / 100
    
    # Add some useful derived metrics
    df['moneyness'] = S / K
    df['intrinsic_value'] = np.where(call_mask, 
                                    np.maximum(S - K, 0), 
                                    np.maximum(K - S, 0))
    
    # Clean up naff columns, return dataframe with greeks in 
    cols_to_drop = ['contractSize']
    existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
    clean_df = df.drop(existing_cols_to_drop, axis=1)

    # reorder cols in sensible way
    df_reordered = clean_df.loc[:, ['ticker','optionType', 'strike', 'expirationDate','volume','impliedVolatility', 'lastPrice', 'change', 'percentChange',
     'delta', 'gamma', 'theta', 'vega', 'rho', 'moneyness',
       'intrinsic_value','openInterest', 'bid', 'ask']]


    return df_reordered


def display_price_chart(prices_df):

    if prices_df.empty:
        st.warning("No price data available.")
        return
    
    # Reset index to get datetime as column
    plot_df = prices_df.reset_index()
    
    # Melt dataframe for Plotly
    melted_df = plot_df.melt(id_vars=['Date'], var_name='Ticker', value_name='Price')
    
    # Create time series chart (keeps all data points)
    fig = px.line(melted_df, x='Date', y='Price', color='Ticker')

    # Format x-axis to show fewer labels but keep all data
    fig.update_xaxes(
        dtick="M1",  # Show every month
        tickformat="%d %b %y",
        fixedrange=True,
        # rangeslider_visible=True,   
        
    )

    # edit config to tidy up chart
    config = {'displayModeBar': False}
    
    fig.update_layout(
        height=400,
        dragmode=False
        )
    
    st.plotly_chart(fig, use_container_width=True, config=config)


def display_price_metrics(prices_df, ticker):

    if prices_df.empty or ticker not in prices_df.columns:
        st.warning(f"No data available for {ticker}")
        return
    
    # Get latest and previous day prices
    ticker_data = prices_df[ticker].dropna()
    if len(ticker_data) < 2:
        st.warning(f"Insufficient data for {ticker}")
        return
    
    current_price = ticker_data.iloc[-1]
    previous_price = ticker_data.iloc[-2]
    
    # Calculate daily change
    daily_change = current_price - previous_price
    daily_change_pct = (daily_change / previous_price) * 100
    
    # Create columns for metrics
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric(
            label=f'{ticker} Price',
            value=f'${current_price:.2f}',
            delta=f'{daily_change:+.2f}'
        )

    st.markdown(
    """
    <style>
    [data-testid="stMetricValue"] {
        font-size: 32px;  /* adjust */
    }
    [data-testid="stMetricDelta"] {
        font-size: 16px;  /* adjust */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

    col2.metric(
        label="Daily Change %",
        value="", 
        delta=f"{daily_change_pct:+.2f}%"
    )