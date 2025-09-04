import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text
import streamlit as st

def get_connection():
    """Return SQLAlchemy engine using DATABASE_URL from Streamlit secrets."""
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(db_url)
    return engine

def init_db():
    """Create the minimal options_cache table if it does not exist."""
    engine = get_connection()
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS options_cache (
                id SERIAL PRIMARY KEY,
                ticker TEXT,
                option_type TEXT,
                strike DOUBLE PRECISION,
                expiration_date DATE,
                last_price DOUBLE PRECISION,
                volume INT,
                stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))

def save_options(df):
    """Save simplified options data into options_cache."""
    if df.empty:
        return False
    
    engine = get_connection()

    # Keep only needed columns and rename
    df_simple = df[['ticker','optionType','strike','expirationDate','lastPrice','volume']].copy()
    df_simple.rename(columns={
        'optionType': 'option_type',
        'expirationDate': 'expiration_date',
        'lastPrice': 'last_price'
    }, inplace=True)

    # Convert types
    df_simple['expiration_date'] = pd.to_datetime(df_simple['expiration_date']).dt.date
    df_simple['stored_at'] = datetime.now()
    df_simple['volume'] = df_simple['volume'].fillna(0).astype(int)

    # Insert
    df_simple.to_sql('options_cache', engine, if_exists='append', index=False)
    return True

def load_options(ticker=None):
    """Load recent options data. If ticker is given, filter."""
    engine = get_connection()
    with engine.connect() as conn:
        if ticker:
            query = text("""
                SELECT * FROM options_cache
                WHERE ticker = :t
                ORDER BY stored_at DESC
            """)
            df = pd.read_sql(query, conn, params={'t': ticker})
        else:
            query = text("""
                SELECT * FROM options_cache
                WHERE stored_at > NOW() - INTERVAL '24 hours'
                ORDER BY stored_at DESC
            """)
            df = pd.read_sql(query, conn)
    return df

def test_connection():
    """Basic connectivity and row count check."""
    try:
        engine = get_connection()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            result = conn.execute(text("SELECT COUNT(*) FROM options_cache"))
            count = result.scalar()
            print(f"✅ Connected. options_cache has {count} rows.")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False