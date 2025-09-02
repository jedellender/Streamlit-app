import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import datetime as dt

# Fetch Data including current price
@st.cache_data(ttl=300)
def fetch_data(tickers):
    prices, infos, latest_price = {}, {}, {} 
    for t in tickers:
        try:
            ticker = yf.Ticker(t)
            hist = ticker.history(period="1y")["Close"]  # 1 year hist price
            prices[t] = hist # 
            info = ticker.info # desc ticker data 
            current_price = info.get('currentPrice')
            infos[t] = {
                "Name": info.get("longName",np.nan),
                "Current Price": f"${current_price:.2f}" if current_price is not None and not np.isnan(current_price) else "N/A",
                "P/B (B)": f"{info.get('priceToBook',np.nan):.2f}x",
                "Sector": info.get('sector',np.nan),
                "MarketCap (B)": info.get('marketCap',np.nan)/1e9,
                "Forward P/E": (info.get('forwardPE',np.nan))
            }
            # get latest price and time 
            last_updated = dt.datetime.now().strftime("%H:%M %d/%m/%y")
            latest_price[t] = {
                'price': hist.iloc[-1],
                'time': last_updated
            }
                        
        except Exception as e:
            print(f"Failed to fetch data for {t}. Reason: {e}")
            prices[t] = pd.Series()  #  Empty series on failure
            infos[t] = {} # Empty info on failure
            latest_price[t] = np.nan

    # Create DataFrames from the dictionaries and return them
    prices_df = pd.DataFrame(prices)
    infos_df = pd.DataFrame(infos).T
    current_prices = pd.DataFrame.from_dict(latest_price, orient='index')

    st.dataframe(current_prices)
    return prices_df, infos_df, current_prices


@st.cache_data(ttl=900)
def fetch_options_data(tickers): # function to process options data
    all_options = []

    for t in tickers:
        ticker = yf.Ticker(t) 
        try: 
            for exp_date in ticker.options: # loop through all exp dates
                try:
                    options = ticker.option_chain(exp_date) # get options 

                    # process calls 
                    calls_df = options.calls.copy()
                    calls_df["optionType"] = "Call"
                    calls_df["expirationDate"] = exp_date
                    calls_df['ticker'] = t

                    puts_df = options.puts.copy()
                    puts_df["optionType"] = "Put"
                    puts_df["expirationDate"] = exp_date
                    puts_df['ticker'] = t

                    all_options.extend([calls_df, puts_df])
                
                except Exception as e:
                    print(f"Failed to fetch data for {t} on {exp_date}. Reason: {e}")

        except Exception as e:
                    print(f"Failed to fetch data for {t}. Reason: {e}")

    # Combine all dfs
    if all_options:
        combined_df = pd.concat(all_options, ignore_index=True)
        return combined_df
    else:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def get_clean_options(options_df, current_prices):

    c_df = options_df.copy()

    c_df['current_price'] = c_df['ticker'].map(dict(current_prices['price']))

    # Convert expiration dates to datetime if they're strings
    if c_df['expirationDate'].dtype == 'object':
        c_df['expirationDate'] = pd.to_datetime(c_df['expirationDate'])

    # Calculate time to expiration in years
    today = pd.Timestamp.now()
    c_df['time_to_expiry'] = (c_df['expirationDate'] - today).dt.days / 365.0
    
    # clean c_df data
    # drop bid, ask and IV etc -> inaccurate
    cols_to_drop = ['bid', 'ask', 'contractSize'] 

    c_df = c_df.drop(columns=[c for c in cols_to_drop if c in c_df.columns], errors='ignore') # del cols iff not alrdy del

    # sort for quality 
    c_df = c_df[(c_df['time_to_expiry'] > 0) & 
            (c_df['current_price'].notna()) & 
            (c_df['volume'] >=50) &
            (c_df['openInterest'] >=200)
    ]

    return c_df
