import psycopg2
import pandas as pd
import streamlit as st
from datetime import datetime

# Put your Supabase connection string here for now (we'll secure it later)
DATABASE_URL = "postgresql://postgres:[YOUR-PASSWORD]@db.xxxx.supabase.co:5432/postgres"

def get_connection():
    """Get database connection"""
    # Try Streamlit secrets first (for deployed app)
    if "DATABASE_URL" in st.secrets:
        return psycopg2.connect(st.secrets["DATABASE_URL"])
    # Otherwise use the hardcoded URL (for testing)
    return psycopg2.connect(DATABASE_URL)

def init_db():
    """Create table if needed"""
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
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
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Database ready")
    except Exception as e:
        print(f"Database init error (might be ok): {e}")

def save_options(options_df):
    """Save options data"""
    if options_df.empty:
        return
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Clear old data
        tickers = options_df['ticker'].unique()
        for ticker in tickers:
            cursor.execute("DELETE FROM options_cache WHERE ticker = %s", (ticker,))
        
        # Save new data
        for _, row in options_df.iterrows():
            cursor.execute("""
                INSERT INTO options_cache 
                (ticker, option_type, strike, expiration_date, last_price, volume, 
                 open_interest, implied_volatility, bid, ask, change_price, percent_change,
                 in_the_money, contract_symbol, last_trade_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                row.get('ticker'),
                row.get('optionType'),
                row.get('strike'),
                pd.to_datetime(row.get('expirationDate')).date() if pd.notna(row.get('expirationDate')) else None,
                row.get('lastPrice'),
                row.get('volume'),
                row.get('openInterest'),
                row.get('impliedVolatility'),
                row.get('bid'),
                row.get('ask'),
                row.get('change'),
                row.get('percentChange'),
                row.get('inTheMoney'),
                row.get('contractSymbol'),
                pd.to_datetime(row.get('lastTradeDate')) if pd.notna(row.get('lastTradeDate')) else None
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Saved {len(options_df)} options")
    except Exception as e:
        print(f"Save error: {e}")

def load_options(tickers):
    """Load options data"""
    try:
        conn = get_connection()
        
        placeholders = ','.join(['%s'] * len(tickers))
        query = f"""
            SELECT * FROM options_cache 
            WHERE ticker IN ({placeholders})
            ORDER BY stored_at DESC
        """
        
        df = pd.read_sql(query, conn, params=tickers)
        conn.close()
        
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
        print(f"Load error: {e}")
        return pd.DataFrame()

# Initialize database when imported
init_db()