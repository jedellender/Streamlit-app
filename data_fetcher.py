import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import datetime as dt

# import local modules
from db import save_options, load_options

# Fetch Data including current price
@st.cache_data(ttl=300)
def fetch_data(tickers):
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
            print(f"Failed to fetch data for {t}. Reason: {e}")
            prices[t] = pd.Series()  
            infos[t] = {}
            latest_price[t] = {"price": np.nan, "time": None}

    prices_df = pd.DataFrame(prices)
    infos_df = pd.DataFrame(infos).T
    current_prices = pd.DataFrame.from_dict(latest_price, orient='index')

    return prices_df, infos_df, current_prices

@st.cache_data(ttl=900)
def fetch_options_data(tickers):
    """Fetch options with robust database fallback"""
    all_options = []
    failed_tickers = []
    
    # Try fetching from yfinance first
    for t in tickers:
        ticker = yf.Ticker(t)
        ticker_success = False
        
        try:
            exp_dates = ticker.options
            if not exp_dates:  # No expiration dates available
                print(f"No options data available for {t} from yfinance")
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
    
    # Combine yfinance data if we got any
    yfinance_df = pd.DataFrame()
    if all_options:
        yfinance_df = pd.concat(all_options, ignore_index=True)
        print(f"Fetched {len(yfinance_df)} options from yfinance for {len(set(yfinance_df['ticker']))} tickers")
        
        # Save to database
        save_success = save_options(yfinance_df)
        if save_success:
            print("Successfully cached yfinance data")
    
    # If we have failed tickers or no data at all, try database
    if failed_tickers or yfinance_df.empty:
        print(f"Attempting to load cached data for: {failed_tickers if failed_tickers else 'all tickers'}")
        
        # Load from database
        db_df = load_options(failed_tickers if failed_tickers else tickers)
        
        if not db_df.empty:
            # Combine yfinance and database data
            if not yfinance_df.empty:
                # Remove any duplicate tickers from db_df that we already got from yfinance
                successful_tickers = set(yfinance_df['ticker'].unique())
                db_df = db_df[~db_df['ticker'].isin(successful_tickers)]
                
                if not db_df.empty:
                    combined_df = pd.concat([yfinance_df, db_df], ignore_index=True)
                    print(f"Combined {len(yfinance_df)} live + {len(db_df)} cached options")
                    return combined_df
                else:
                    return yfinance_df
            else:
                print(f"Using {len(db_df)} cached options (market likely closed)")
                return db_df
    
    # Return whatever we have
    if not yfinance_df.empty:
        return yfinance_df
    else:
        print("No options data available from any source")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_clean_options(options_df, current_prices):
    """Clean options data with better error handling"""
    if options_df.empty:
        return pd.DataFrame()
    
    c_df = options_df.copy()
    
    # Map current prices
    c_df['current_price'] = c_df['ticker'].map(dict(current_prices['price']))
    
    # Handle date conversion more robustly
    if 'expirationDate' in c_df.columns:
        c_df['expirationDate'] = pd.to_datetime(c_df['expirationDate'], errors='coerce')
    
    # Calculate days to expiry
    today = pd.Timestamp.now()
    c_df['days_to_expiry'] = (c_df['expirationDate'] - today).dt.days
    c_df['time_to_expiry'] = c_df['days_to_expiry'] / 365.0
    
    # Drop unnecessary columns
    cols_to_drop = ['bid', 'ask', 'contractSize']
    c_df = c_df.drop(columns=[c for c in cols_to_drop if c in c_df.columns], errors='ignore')
    
    # Filter for quality - be more lenient with cached data
    c_df = c_df[
        (c_df['time_to_expiry'] > 0) & 
        (c_df['current_price'].notna()) & 
        (c_df['volume'] >= 10) &  # Lower threshold for cached data
        (c_df['openInterest'] >= 50)  # Lower threshold for cached data
    ]
    
    return c_df