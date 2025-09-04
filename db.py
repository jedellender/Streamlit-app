import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import streamlit as st

def get_connection():
    """Return SQLAlchemy engine using DATABASE_URL from Streamlit secrets."""
    db_url = st.secrets["DATABASE_URL"]
    engine = create_engine(db_url, pool_pre_ping=True)
    return engine

def init_db():
    """Ensure the options_cache table exists with proper schema."""
    engine = get_connection()
    try:
        with engine.begin() as conn:
            # Check if table exists and has correct schema
            result = conn.execute(text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'options_cache'
            """))
            existing_columns = [row[0] for row in result.fetchall()]
            
            # If table doesn't exist or has wrong schema, create/recreate
            if 'option_type' not in existing_columns:
                print("🔄 Table schema outdated, please run migration script first")
                return False
            
            print("✅ Database schema is correct")
            return True
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False

def cleanup_expired_options():
    """Remove options that expired more than 7 days ago."""
    engine = get_connection()
    try:
        with engine.begin() as conn:
            cutoff_date = datetime.now().date() - timedelta(days=7)
            result = conn.execute(
                text("DELETE FROM options_cache WHERE expiration_date < :cutoff"),
                {"cutoff": cutoff_date}
            )
            deleted_count = result.rowcount
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} expired options")
            return deleted_count
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")
        return 0

def save_options(df):
    """Save options data with enhanced schema support."""
    if df.empty:
        print("⚠️ No options data to save")
        return False
    
    try:
        # Map yfinance columns to database columns
        column_mapping = {
            'ticker': 'ticker',
            'optionType': 'option_type', 
            'strike': 'strike',
            'expirationDate': 'expiration_date',
            'lastPrice': 'last_price',
            'volume': 'volume',
            'contractSymbol': 'contract_symbol',
            'openInterest': 'open_interest', 
            'impliedVolatility': 'implied_volatility',
            'inTheMoney': 'in_the_money',
            'lastTradeDate': 'last_trade_date'
        }
        
        # Select and rename columns that exist
        available_cols = [col for col in column_mapping.keys() if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # Rename columns
        df_clean.rename(columns={k: v for k, v in column_mapping.items() if k in available_cols}, inplace=True)
        
        # Data validation and cleaning
        df_clean = df_clean.dropna(subset=['ticker', 'option_type', 'strike', 'expiration_date'])
        df_clean = df_clean[df_clean['strike'] > 0]
        
        if df_clean.empty:
            print("⚠️ No valid options data after cleaning")
            return False
        
        # Convert data types
        df_clean['expiration_date'] = pd.to_datetime(df_clean['expiration_date']).dt.date
        df_clean['stored_at'] = datetime.now()
        
        # Handle numeric columns safely
        numeric_cols = ['volume', 'open_interest']
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype(int)
        
        if 'last_price' in df_clean.columns:
            df_clean['last_price'] = pd.to_numeric(df_clean['last_price'], errors='coerce').fillna(0.0)
        
        if 'implied_volatility' in df_clean.columns:
            df_clean['implied_volatility'] = pd.to_numeric(df_clean['implied_volatility'], errors='coerce')
        
        # Handle last_trade_date
        if 'last_trade_date' in df_clean.columns:
            df_clean['last_trade_date'] = pd.to_datetime(df_clean['last_trade_date'], errors='coerce')
        
        engine = get_connection()
        
        # Clean up existing data for these tickers to avoid duplicates
        tickers = df_clean['ticker'].unique()
        with engine.begin() as conn:
            for ticker in tickers:
                conn.execute(
                    text("DELETE FROM options_cache WHERE ticker = :ticker"),
                    {"ticker": ticker}
                )
        
        # Insert new data
        rows_inserted = len(df_clean)
        df_clean.to_sql('options_cache', engine, if_exists='append', index=False)
        
        print(f"✅ Saved {rows_inserted} options for {len(tickers)} tickers")
        return True
        
    except Exception as e:
        print(f"❌ Failed to save options: {e}")
        return False

def load_options(tickers=None):
    """Load recent options data, returning format compatible with existing code."""
    engine = get_connection()
    try:
        with engine.connect() as conn:
            if tickers:
                if isinstance(tickers, str):
                    tickers = [tickers]
                
                placeholders = ', '.join([f':ticker_{i}' for i in range(len(tickers))])
                params = {f'ticker_{i}': ticker for i, ticker in enumerate(tickers)}
                
                query = text(f"""
                    SELECT ticker, option_type as optionType, strike, expiration_date as expirationDate,
                           last_price as lastPrice, volume, contract_symbol as contractSymbol,
                           open_interest as openInterest, implied_volatility as impliedVolatility,
                           in_the_money as inTheMoney, last_trade_date as lastTradeDate,
                           'USD' as currency, stored_at
                    FROM options_cache
                    WHERE ticker IN ({placeholders})
                      AND expiration_date > CURRENT_DATE
                      AND stored_at > NOW() - INTERVAL '48 hours'
                    ORDER BY ticker, expiration_date, strike
                """)
                df = pd.read_sql(query, conn, params=params)
            else:
                query = text("""
                    SELECT ticker, option_type as optionType, strike, expiration_date as expirationDate,
                           last_price as lastPrice, volume, contract_symbol as contractSymbol,
                           open_interest as openInterest, implied_volatility as impliedVolatility,
                           in_the_money as inTheMoney, last_trade_date as lastTradeDate,
                           'USD' as currency, stored_at
                    FROM options_cache
                    WHERE expiration_date > CURRENT_DATE
                      AND stored_at > NOW() - INTERVAL '24 hours'
                    ORDER BY ticker, expiration_date, strike
                """)
                df = pd.read_sql(query, conn)
            
            if not df.empty:
                print(f"📖 Loaded {len(df)} cached options for {df['ticker'].nunique()} tickers")
            
            return df
            
    except Exception as e:
        print(f"❌ Failed to load options: {e}")
        return pd.DataFrame()

def get_database_stats():
    """Get database statistics for monitoring."""
    engine = get_connection()
    try:
        with engine.connect() as conn:
            stats = conn.execute(text("""
                SELECT 
                    COUNT(*) as total_options,
                    COUNT(DISTINCT ticker) as unique_tickers,
                    MIN(expiration_date) as earliest_expiry,
                    MAX(expiration_date) as latest_expiry,
                    MIN(stored_at) as oldest_data,
                    MAX(stored_at) as newest_data,
                    COUNT(CASE WHEN expiration_date < CURRENT_DATE THEN 1 END) as expired_options
                FROM options_cache
            """)).fetchone()
            
            return {
                'total_options': stats[0],
                'unique_tickers': stats[1], 
                'earliest_expiry': stats[2],
                'latest_expiry': stats[3],
                'oldest_data': stats[4],
                'newest_data': stats[5],
                'expired_options': stats[6]
            }
    except Exception as e:
        print(f"❌ Failed to get stats: {e}")
        return {}

def test_connection():
    """Test database connectivity and schema."""
    try:
        engine = get_connection()
        with engine.connect() as conn:
            # Test basic connectivity
            conn.execute(text("SELECT 1"))
            
            # Check schema
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'options_cache'
                ORDER BY ordinal_position
            """))
            columns = result.fetchall()
            
            # Get row count
            count_result = conn.execute(text("SELECT COUNT(*) FROM options_cache"))
            row_count = count_result.scalar()
            
            print(f"✅ Database connected successfully")
            print(f"📊 Table has {len(columns)} columns and {row_count} rows")
            
            return True
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False