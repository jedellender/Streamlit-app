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
    if options_df.empty:
        print("Received an empty DataFrame. No data to save.")
        return

    engine = get_connection()
    table_name = 'options_cache'
    
    # 1. Prepare a copy of the DataFrame to match the database schema
    df_to_save = options_df.copy()

    # Map yfinance's camelCase column names to the database's snake_case names
    rename_map = {
        'optionType': 'option_type',
        'expirationDate': 'expiration_date',
        'lastPrice': 'last_price',
        'openInterest': 'open_interest',
        'impliedVolatility': 'implied_volatility',
        'change': 'change_price',
        'percentChange': 'percent_change',
        'inTheMoney': 'in_the_money',
        'contractSymbol': 'contract_symbol',
        'lastTradeDate': 'last_trade_date'
    }
    df_to_save.rename(columns=rename_map, inplace=True)
    
    # Ensure datetime columns are in the correct format
    df_to_save['expiration_date'] = pd.to_datetime(df_to_save['expiration_date']).dt.date
    df_to_save['last_trade_date'] = pd.to_datetime(df_to_save['last_trade_date'])

    try:
        with engine.begin() as conn:
            # 2. Optimize deletion: Delete all relevant tickers in a single command
            tickers_to_update = tuple(df_to_save['ticker'].unique())
            if tickers_to_update:
                conn.execute(
                    text("DELETE FROM options_cache WHERE ticker IN :tickers"),
                    {"tickers": tickers_to_update}
                )

            # 3. Perform the bulk insert
            df_to_save.to_sql(
                name=table_name,
                con=conn,
                if_exists='append',  # Append data since we already cleared old records
                index=False,         # Do not write the DataFrame index as a column
                method='multi'       # Use multi-value INSERTs for efficiency
            )
            print(f"✅ Bulk saved {len(df_to_save)} options to the database.")

    except Exception as e:
        print(f"❌ Bulk save error: {e}")


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