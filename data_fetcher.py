import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import datetime as dt

@st.cache_data(ttl=300)  # 5 minute cache for price data
def fetch_data(tickers):
    """Fetch equity price and info data."""
    prices, infos, latest_price = {}, {}, {} 
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="1y")["Close"]
            prices[t] = hist
            info = ticker.info
            current_price = info.get('currentPrice')
            infos[t] = {
                "Name": info.get("longName", np.nan),
                "Current Price": f"${current_price:.2f}" if current_price is not None and not np.isnan(current_price) else "N/A",
                "P/B (B)": f"{info.get('priceToBook', np.nan):.2f}x",
                "Sector": info.get('sector', np.nan),
                "MarketCap (B)": info.get('marketCap', np.nan)/1e9 if info.get('marketCap') else np.nan,
                "Forward P/E": info.get('forwardPE', np.nan)
            }
            # Get latest price and time
            last_updated = dt.datetime.now().strftime("%H:%M %d/%m/%y")
            latest_price[t] = {
                'price': hist.iloc[-1] if not hist.empty else np.nan,
                'time': last_updated
            }
                        
        except Exception as e:
            print(f"Failed to fetch data for {t}: {e}")
            prices[t] = pd.Series()  
            infos[t] = {}
            latest_price[t] = {"price": np.nan, "time": None}

    prices_df = pd.DataFrame(prices)
    infos_df = pd.DataFrame(infos).T
    current_prices = pd.DataFrame.from_dict(latest_price, orient='index')
    current_prices['price'] = pd.to_numeric(current_prices['price'], errors='coerce')
    current_prices['time'] = pd.to_datetime(current_prices['time'], errors='coerce')

    return prices_df, infos_df, current_prices

@st.cache_data(ttl=900)  # 15 minute cache for options - reasonable for options data
def fetch_options_data(tickers):
    """Fetch options data from yfinance only."""
    all_options = []
    failed_tickers = []
    
    for t in tickers:
        ticker = yf.Ticker(t)
        ticker_success = False
        
        try:
            exp_dates = ticker.options
            if not exp_dates:
                print(f"No options available for {t}")
                failed_tickers.append(t)
                continue
                
            for exp_date in exp_dates:
                try:
                    options = ticker.option_chain(exp_date)
                    
                    # Process calls
                    if hasattr(options, 'calls') and not options.calls.empty:
                        calls_df = options.calls.copy()
                        calls_df["optionType"] = "Call"
                        calls_df["expirationDate"] = exp_date
                        calls_df['ticker'] = t
                        all_options.append(calls_df)
                        ticker_success = True
                    
                    # Process puts
                    if hasattr(options, 'puts') and not options.puts.empty:
                        puts_df = options.puts.copy()
                        puts_df["optionType"] = "Put"
                        puts_df["expirationDate"] = exp_date
                        puts_df['ticker'] = t
                        all_options.append(puts_df)
                        ticker_success = True
                        
                except Exception as e:
                    print(f"Failed to fetch {t} options for {exp_date}: {e}")
            
            if not ticker_success:
                failed_tickers.append(t)
                
        except Exception as e:
            print(f"Failed to get options for {t}: {e}")
            failed_tickers.append(t)
    
    if all_options:
        result_df = pd.concat(all_options, ignore_index=True)
        print(f"✅ Fetched {len(result_df)} options for {len(set(result_df['ticker']))} tickers")
        
        if failed_tickers:
            print(f"⚠️ Failed to fetch options for: {failed_tickers}")
        
        return result_df
    else:
        print(f"❌ No options data available. Failed tickers: {failed_tickers}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_clean_options(options_df, current_prices, t_max, v_max=2, on=False):
    """Clean and filter options data."""
    if options_df.empty:
        return pd.DataFrame()
    
    c_df = options_df.copy()
    
    # Map current prices
    c_df['current_price'] = c_df['ticker'].map(dict(current_prices['price']))
    
    # Handle date conversion
    if 'expirationDate' in c_df.columns:
        c_df['expirationDate'] = pd.to_datetime(c_df['expirationDate'], errors='coerce')
    
    # Calculate days to expiry
    today = pd.Timestamp.now()
    c_df['days_to_expiry'] = (c_df['expirationDate'] - today).dt.days
    c_df['time_to_expiry'] = c_df['days_to_expiry'] / 365.0

    # add bid ask spread and pct
    c_df['bid_ask_spread'] = c_df['ask'] - c_df['bid']
    c_df['bid_ask_spread_pct'] = (c_df['bid_ask_spread'] / c_df['lastPrice']) * 100
    
    
    # Drop unnecessary columns
    cols_to_drop = ['contractSize']
    
    c_df = c_df.drop(columns=[c for c in cols_to_drop if c in c_df.columns], errors='ignore')
    
    # Filter for quality data
    c_df = c_df[
        (c_df['current_price'].notna() | on) &  # Skip if using backup data  
        (c_df['volume'] >= 50) &  # Higher threshold since no fallback
        (c_df['openInterest'] >= 100) &  # Higher threshold for quality
        (c_df['time_to_expiry'] > 0) & 
        (0.01 < c_df['impliedVolatility']) &
        (c_df['impliedVolatility'] < v_max)  &
        (c_df['bid_ask_spread_pct'] < 20)
        ]   
    
    return c_df