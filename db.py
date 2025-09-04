import pandas as pd
import streamlit as st
from datetime import datetime
from sqlalchemy import create_engine, text

def get_connection():
    DATABASE_URL = st.secrets["DATABASE_URL"]
    engine = create_engine(DATABASE_URL)
    return engine

def init_db():
    """Create table if needed"""
    try:
        engine = get_connection()
        with engine.begin() as conn:  # automatically commits
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS options_cache (
                    id SERIAL PRIMARY KEY,
                    ticker VARCHAR(10),
                    option_type VARCHAR(10),
                    strike FLOAT,
                    expiration_date DATE,
                    last_price FLOAT,
                    volume INT,
                    open_interest INT,
                    implied_volatility FLOAT,
                    bid FLOAT,
                    ask FLOAT,
                    change_price FLOAT,
                    percent_change FLOAT,
                    in_the_money BOOLEAN,
                    contract_symbol VARCHAR(50),
                    last_trade_date TIMESTAMP,
                    stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        print("✅ Database ready")
    except Exception as e:
        print(f"Database init error (might be ok): {e}")

def save_options(options_df):
    """Save options data using SQLAlchemy"""
    if options_df.empty:
        return 

    try:
        engine = get_connection()
        tickers = options_df['ticker'].unique()

        with engine.begin() as conn:  # handles commit automatically
            # Clear old data per ticker
            for ticker in tickers:
                conn.execute(
                    text("DELETE FROM options_cache WHERE ticker = :ticker"),
                    {"ticker": ticker}
                )
    
            # Insert new rows
            for _, row in options_df.iterrows():
                conn.execute(
                    text("""
                        INSERT INTO options_cache 
                        (ticker, option_type, strike, expiration_date, last_price, volume, 
                         open_interest, implied_volatility, bid, ask, change_price, percent_change,
                         in_the_money, contract_symbol, last_trade_date)
                        VALUES 
                        (:ticker, :option_type, :strike, :expiration_date, :last_price, :volume,
                         :open_interest, :implied_volatility, :bid, :ask, :change_price, :percent_change,
                         :in_the_money, :contract_symbol, :last_trade_date)
                    """),
                    {
                        "ticker": row.get('ticker'),
                        "option_type": row.get('optionType'),
                        "strike": row.get('strike'),
                        "expiration_date": pd.to_datetime(row.get('expirationDate')).date() 
                                            if pd.notna(row.get('expirationDate')) else None,
                        "last_price": row.get('lastPrice'),
                        "volume": row.get('volume'),
                        "open_interest": row.get('openInterest'),
                        "implied_volatility": row.get('impliedVolatility'),
                        "bid": row.get('bid'),
                        "ask": row.get('ask'),
                        "change_price": row.get('change'),
                        "percent_change": row.get('percentChange'),
                        "in_the_money": row.get('inTheMoney'),
                        "contract_symbol": row.get('contractSymbol'),
                        "last_trade_date": pd.to_datetime(row.get('lastTradeDate')) 
                                            if pd.notna(row.get('lastTradeDate')) else None
                    }
                )
        print(f"✅ Saved {len(options_df)} options")
    except Exception as e:
        print(f"❌ Save error: {e}")


def load_options(tickers):
    """Load options data using SQLAlchemy"""
    if not tickers:
        return pd.DataFrame()

    try:
        engine = get_connection()
        placeholders = ", ".join([":t{}".format(i) for i in range(len(tickers))])
        params = {f"t{i}": t for i, t in enumerate(tickers)}

        query = f"""
            SELECT * FROM options_cache
            WHERE ticker IN ({placeholders})
            ORDER BY stored_at DESC
        """

        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn, params=params)

        # Rename columns back to yfinance format
        if not df.empty:
            df = df.rename(columns={
                'option_type': 'optionType',
                'expiration_date': 'expirationDate',
                'last_price': 'lastPrice',
                'open_interest': 'openInterest',
                'implied_volatility': 'impliedVolatility',
                'change_price': 'change',
                'percent_change': 'percentChange',
                'in_the_money': 'inTheMoney',
                'contract_symbol': 'contractSymbol',
                'last_trade_date': 'lastTradeDate'
            })

        return df
    except Exception as e:
        print(f"❌ Load error: {e}")
        return pd.DataFrame()
