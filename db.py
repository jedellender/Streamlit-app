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
        with engine.begin() as conn:
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
                    currency VARCHAR(10),
                    stored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(ticker, option_type, strike, expiration_date)
                )
            """))
            print("✅ Database ready")
    except Exception as e:
        print(f"Database init error: {e}")

def save_options(options_df):
    """Save options data with proper error handling and column mapping"""
    if options_df.empty:
        print("No data to save - DataFrame is empty")
        return False
    
    engine = get_connection()
    
    try:
        # Prepare DataFrame
        df_to_save = options_df.copy()
        
        # Map columns
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
        
        # Convert date columns
        df_to_save['expiration_date'] = pd.to_datetime(df_to_save['expiration_date']).dt.date
        if 'last_trade_date' in df_to_save.columns:
            df_to_save['last_trade_date'] = pd.to_datetime(df_to_save['last_trade_date'], errors='coerce')
        
        # Add timestamp
        df_to_save['stored_at'] = datetime.now()
        
        # Select only columns that exist in the database
        db_columns = ['ticker', 'option_type', 'strike', 'expiration_date', 'last_price', 
                      'volume', 'open_interest', 'implied_volatility', 'bid', 'ask', 
                      'change_price', 'percent_change', 'in_the_money', 'contract_symbol', 
                      'last_trade_date', 'currency', 'stored_at']
        
        # Only keep columns that exist in both DataFrame and database schema
        columns_to_save = [col for col in db_columns if col in df_to_save.columns]
        df_to_save = df_to_save[columns_to_save]
        
        with engine.begin() as conn:
            # Delete existing data for these tickers
            unique_tickers = df_to_save['ticker'].unique().tolist()
            if unique_tickers:
                # Use proper parameterised query
                placeholders = ', '.join([f':ticker_{i}' for i in range(len(unique_tickers))])
                params = {f'ticker_{i}': ticker for i, ticker in enumerate(unique_tickers)}
                
                delete_query = text(f"DELETE FROM options_cache WHERE ticker IN ({placeholders})")
                conn.execute(delete_query, params)
                print(f"Cleared old data for tickers: {unique_tickers}")
            
            # Insert new data
            df_to_save.to_sql(
                name='options_cache',
                con=conn,
                if_exists='append',
                index=False,
                method='multi'
            )
            print(f"✅ Saved {len(df_to_save)} options to database")
            return True
            
    except Exception as e:
        print(f"❌ Save error: {e}")
        print(f"DataFrame columns: {df_to_save.columns.tolist()}")
        return False

def load_options(tickers=None):
    """Load options with better error handling"""
    try:
        engine = get_connection()
        
        if tickers:
            # Load specific tickers
            placeholders = ', '.join([f':ticker_{i}' for i in range(len(tickers))])
            params = {f'ticker_{i}': ticker for i, ticker in enumerate(tickers)}
            
            query = text(f"""
                SELECT * FROM options_cache
                WHERE ticker IN ({placeholders})
                ORDER BY stored_at DESC
            """)
        else:
            # Load all recent data
            query = text("""
                SELECT * FROM options_cache
                WHERE stored_at > NOW() - INTERVAL '24 hours'
                ORDER BY stored_at DESC
            """)
            params = {}
        
        with engine.connect() as conn:
            df = pd.read_sql(query, conn, params=params if tickers else None)
        
        if not df.empty:
            # Rename columns back to yfinance format
            rename_map = {
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
            }
            df.rename(columns=rename_map, inplace=True)
            print(f"✅ Loaded {len(df)} options from database")
        else:
            print("No cached data found")
            
        return df
        
    except Exception as e:
        print(f"❌ Load error: {e}")
        return pd.DataFrame()

def test_connection():
    """Test database connection and operations"""
    try:
        engine = get_connection()
        with engine.connect() as conn:
            # Test connection
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connected")
            
            # Check table exists
            result = conn.execute(text("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = 'options_cache'
            """))
            if result.scalar() > 0:
                # Count rows
                result = conn.execute(text("SELECT COUNT(*) FROM options_cache"))
                count = result.scalar()
                print(f"✅ Table exists with {count} rows")
                
                # Show recent data
                result = conn.execute(text("""
                    SELECT ticker, COUNT(*) as option_count, MAX(stored_at) as last_update
                    FROM options_cache
                    GROUP BY ticker
                    ORDER BY last_update DESC
                    LIMIT 5
                """))
                recent = result.fetchall()
                if recent:
                    print("\nRecent data:")
                    for row in recent:
                        print(f"  {row[0]}: {row[1]} options, last updated {row[2]}")
            else:
                print("⚠️ Table doesn't exist - run init_db()")
                
            return True
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False