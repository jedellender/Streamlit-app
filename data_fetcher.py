import streamlit as st
import yfinance as yf
import pandas as pd
from db import save_options_overwrite

@st.cache_data(ttl=300)
def fetch_equity_data(tickers):
    """Fetches historical prices and company info for the given tickers."""
    prices, infos = {}, {}
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="1y")["Close"]
            if not hist.empty:
                prices[t] = hist
                infos[t] = ticker.info
        except Exception:
            st.warning(f"Could not fetch equity data for {t}.")

    if not prices:
        return pd.DataFrame(), pd.DataFrame(), {}

    prices_df = pd.DataFrame(prices)
    
    factor_data = {
        t: {
            "MarketCap (B)": infos.get(t, {}).get('marketCap', 0) / 1e9,
            "Forward P/E": infos.get(t, {}).get('forwardPE', None)
        } for t in tickers
    }
    factor_df = pd.DataFrame.from_dict(factor_data, orient='index')
    current_prices = {t: prices_df[t].iloc[-1] for t in prices_df.columns}

    return prices_df, factor_df, current_prices

def fetch_options_data(tickers):
    """
    Fetches live options data from yfinance. If successful, it overwrites
    the database cache with this fresh data.
    """
    all_options = []
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            for exp_date in ticker.options:
                opt_chain = ticker.option_chain(exp_date)
                for opt_type in ['calls', 'puts']:
                    df = getattr(opt_chain, opt_type)
                    if not df.empty:
                        df = df.copy()
                        df['optionType'] = opt_type.capitalize()
                        df['expirationDate'] = exp_date
                        df['ticker'] = t
                        all_options.append(df)
        except Exception:
            st.warning(f"No options data found for {t}. Market may be closed.")

    if not all_options:
        return pd.DataFrame()

    yfinance_df = pd.concat(all_options, ignore_index=True)
    save_options_overwrite(yfinance_df)
    return yfinance_df

def get_clean_options(options_df, current_prices):
    """Cleans and prepares the options DataFrame for analysis."""
    if options_df.empty:
        return pd.DataFrame()
    
    c_df = options_df.copy()
    c_df['current_price'] = c_df['ticker'].map(current_prices)
    c_df['expirationDate'] = pd.to_datetime(c_df['expirationDate'], errors='coerce')
    c_df['time_to_expiry'] = (c_df['expirationDate'] - pd.Timestamp.now()).dt.days / 365.0
    
    c_df.dropna(
        subset=['current_price', 'strike', 'time_to_expiry', 'impliedVolatility'],
        inplace=True
    )
    c_df = c_df[c_df['time_to_expiry'] > 0]
    return c_df

