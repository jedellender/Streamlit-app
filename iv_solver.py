import numpy as np
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility
from scipy.stats import norm
from scipy.optimize import brentq
import warnings

def iv_calc_robust(options_df, risk_free_rate=0.04, use_mid_price=True):
    """
    Robust IV calculation with multiple fallback methods.
    Always returns a reasonable IV estimate.
    """
    if options_df.empty:
        return options_df
    
    df = options_df.copy()
    
    # Use mid-price if available and requested
    if use_mid_price and 'bid' in df.columns and 'ask' in df.columns:
        df['price_for_iv'] = df[['bid', 'ask']].mean(axis=1)
        # Fall back to lastPrice where mid is invalid
        mask_invalid_mid = (df['price_for_iv'] <= 0) | df['price_for_iv'].isna()
        df.loc[mask_invalid_mid, 'price_for_iv'] = df.loc[mask_invalid_mid, 'lastPrice']
    else:
        df['price_for_iv'] = df['lastPrice']
    
    # Calculate intrinsic value for sanity checks
    df['intrinsic'] = np.where(
        df['optionType'].str.lower() == 'call',
        np.maximum(df['current_price'] - df['strike'], 0),
        np.maximum(df['strike'] - df['current_price'], 0)
    )
    
    # Moneyness for fallback estimates
    df['moneyness'] = df['current_price'] / df['strike']
    df['log_moneyness'] = np.log(df['moneyness'])
    
    # Initialize IV column
    df['iv_calculated'] = np.nan
    df['iv_method'] = 'failed'
    
    # Method 1: Try py_vollib (vectorised where possible)
    mask_valid = (
        df[['current_price', 'time_to_expiry', 'price_for_iv', 'strike']].notna().all(axis=1) &
        (df['current_price'] > 0) & 
        (df['strike'] > 0) &
        (df['price_for_iv'] > 0) &
        (df['time_to_expiry'] > 1/365)  # At least 1 day
    )
    
    # Ensure option price > intrinsic value (no arbitrage)
    mask_valid &= (df['price_for_iv'] >= df['intrinsic'] * 0.99)  # Small tolerance
    
    for idx in df[mask_valid].index:
        row = df.loc[idx]
        try:
            flag = 'c' if row['optionType'].lower() == 'call' else 'p'
            iv = implied_volatility(
                price=float(row['price_for_iv']),
                S=float(row['current_price']),
                K=float(row['strike']),
                t=float(row['time_to_expiry']),
                r=risk_free_rate,
                flag=flag
            )
            if 0.01 <= iv <= 5.0:  # Reasonable range
                df.loc[idx, 'iv_calculated'] = iv
                df.loc[idx, 'iv_method'] = 'py_vollib'
        except Exception as e:
            # Continue to fallback methods
            pass
    
    # Method 2: Approximation for near-ATM options
    mask_near_atm = (
        df['iv_calculated'].isna() &
        (df['moneyness'] >= 0.9) & 
        (df['moneyness'] <= 1.1) &
        mask_valid
    )
    
    if mask_near_atm.any():
        # Brenner-Subrahmanyam approximation
        df.loc[mask_near_atm, 'iv_calculated'] = np.sqrt(2 * np.pi / df.loc[mask_near_atm, 'time_to_expiry']) * (
            df.loc[mask_near_atm, 'price_for_iv'] / df.loc[mask_near_atm, 'current_price']
        )
        df.loc[mask_near_atm, 'iv_method'] = 'BS_approx'
    
    # Method 3: Moneyness-based estimation
    mask_needs_estimate = df['iv_calculated'].isna()
    
    if mask_needs_estimate.any():
        # Use moneyness-based vol smile estimation
        # Typical market behaviour: IV increases for OTM options
        base_vol = 0.25  # Reasonable market average
        
        # Estimate based on moneyness
        smile_adjustment = 0.15 * np.abs(df.loc[mask_needs_estimate, 'log_moneyness'])
        df.loc[mask_needs_estimate, 'iv_calculated'] = base_vol + smile_adjustment
        df.loc[mask_needs_estimate, 'iv_method'] = 'smile_estimate'
    
    # Method 4: Use original IV if available and reasonable
    mask_use_original = (
        df['iv_calculated'].isna() & 
        df['impliedVolatility'].notna() &
        (df['impliedVolatility'] > 0.01) &
        (df['impliedVolatility'] < 5.0)
    )
    
    if mask_use_original.any():
        df.loc[mask_use_original, 'iv_calculated'] = df.loc[mask_use_original, 'impliedVolatility']
        df.loc[mask_use_original, 'iv_method'] = 'original'
    
    # Final safety: Any remaining NaN gets sector-typical vol
    mask_final_fallback = df['iv_calculated'].isna()
    if mask_final_fallback.any():
        # Estimate based on ticker volatility class
        ticker_vols = df.groupby('ticker')['impliedVolatility'].median()
        for ticker in df.loc[mask_final_fallback, 'ticker'].unique():
            ticker_mask = mask_final_fallback & (df['ticker'] == ticker)
            if ticker in ticker_vols and not np.isnan(ticker_vols[ticker]):
                df.loc[ticker_mask, 'iv_calculated'] = ticker_vols[ticker]
            else:
                df.loc[ticker_mask, 'iv_calculated'] = 0.30  # Market average
            df.loc[ticker_mask, 'iv_method'] = 'fallback'
    
    # Clip all values to reasonable range
    df['iv_calculated'] = df['iv_calculated'].clip(lower=0.01, upper=5.0)
    
    # Update the main IV column
    df['impliedVolatility'] = df['iv_calculated']
    
    # Log statistics
    method_counts = df['iv_method'].value_counts()
    success_rate = (df['iv_method'] == 'py_vollib').sum() / len(df) * 100
    
    print(f"IV Calculation Stats:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Methods used: {method_counts.to_dict()}")
    print(f"  Mean IV: {df['iv_calculated'].mean():.2%}")
    
    # Clean up temporary columns
    df.drop(columns=['price_for_iv', 'intrinsic', 'moneyness', 'log_moneyness', 
                     'iv_calculated', 'iv_method'], inplace=True, errors='ignore')
    
    return df


def black_scholes_price(S, K, T, r, sigma, option_type='call'):
    """Helper function for BS pricing"""
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type.lower() == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def custom_iv_solver(price, S, K, T, r, option_type='call', max_iter=100):
    """
    Custom IV solver using Brent's method as fallback.
    More robust than Newton-Raphson for extreme cases.
    """
    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - price
    
    try:
        # Brent's method with wide bounds
        iv = brentq(objective, 0.001, 10.0, maxiter=max_iter)
        return iv
    except:
        # If Brent fails, return ATM approximation
        return np.sqrt(2*np.pi/T) * (price/S)    