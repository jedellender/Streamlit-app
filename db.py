import pandas as pd
from sqlalchemy import create_engine, text
import streamlit as st

# Use st.cache_resource to create a single, persistent connection engine.
@st.cache_resource
def get_connection():
    """Return a cached SQLAlchemy engine using DATABASE_URL from Streamlit secrets."""
    try:
        db_url = st.secrets["DATABASE_URL"]
        return create_engine(db_url)
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return None

def init_db():
    """Create the options_cache table with a more robust schema if it does not exist."""
    engine = get_connection()
    if not engine:
        return

    try:
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS options_cache (
                    contractSymbol TEXT PRIMARY KEY,
                    ticker TEXT,
                    optionType TEXT,
                    strike DOUBLE PRECISION,
                    expirationDate DATE,
                    lastPrice DOUBLE PRECISION,
                    volume DOUBLE PRECISION,
                    openInterest DOUBLE PRECISION,
                    impliedVolatility DOUBLE PRECISION,
                    inTheMoney BOOLEAN,
                    lastTradeDate TIMESTAMP,
                    stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """))
        print("Database initialized.")
    except Exception as e:
        # This error is often benign if the table already exists with a different schema.
        print(f"Database init warning: {e}")


def save_options_overwrite(df):
    """
    Deletes all existing data and saves the new, full DataFrame.
    This ensures the cache is always a fresh, complete snapshot.
    """
    if df.empty:
        print("Received empty DataFrame, skipping database save.")
        return False

    engine = get_connection()
    if not engine:
        return False

    # Define the columns that match the database schema
    db_columns = [
        'contractSymbol', 'ticker', 'optionType', 'strike', 'expirationDate',
        'lastPrice', 'volume', 'openInterest', 'impliedVolatility', 'inTheMoney', 'lastTradeDate'
    ]

    # Prepare a copy with only the necessary columns to avoid errors
    df_to_save = df[[col for col in db_columns if col in df.columns]].copy()
    
    # Ensure data types are correct for the database write
    df_to_save['expirationDate'] = pd.to_datetime(df_to_save['expirationDate']).dt.date
    df_to_save['lastTradeDate'] = pd.to_datetime(df_to_save['lastTradeDate'])

    try:
        with engine.begin() as conn:
            # 1. Delete all old data from the table
            conn.execute(text("DELETE FROM options_cache"))
            
            # 2. Insert the new, complete dataset
            df_to_save.to_sql('options_cache', conn, if_exists='append', index=False)
            
            st.toast(f"Saved {len(df_to_save)} options to cache.")
        return True
    except Exception as e:
        st.error(f"Failed to save to database: {e}")
        return False

def load_options_from_cache():
    """Loads the entire options cache from the database."""
    engine = get_connection()
    if not engine:
        return pd.DataFrame()

    try:
        with engine.connect() as conn:
            df = pd.read_sql("SELECT * FROM options_cache", conn)
        return df
    except Exception as e:
        st.error(f"Failed to load from database: {e}")
        return pd.DataFrame()

