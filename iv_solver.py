import numpy as np
import pandas as pd
from py_vollib.black_scholes.implied_volatility import implied_volatility
from py_vollib.black_scholes import black_scholes

def iv_calc(options_df, risk_free_rate=0.04):
    """
    Replace yfinance IV with correct IV using py_vollib
    """
    df = options_df.copy()
    df['correct_iv'] = np.nan
    
    for idx, row in df.iterrows():
        try:
            # Convert option type to py_vollib format
            flag = 'c' if row['optionType'].lower() == 'call' else 'p'
            
            # Calculate IV using py_vollib
            iv = implied_volatility(
                price=row['lastPrice'],
                S=row['current_price'], 
                K=row['strike'],
                t=row['time_to_expiry'],
                r=risk_free_rate,
                flag=flag
            )
            
            df.at[idx, 'correct_iv'] = iv
            
        except:
            # Keep original if calculation fails
            df.at[idx, 'correct_iv'] = row['impliedVolatility']
    
    # Replace the impliedVolatility column
    df['impliedVolatility'] = df['correct_iv']
    df.drop('correct_iv', axis=1, inplace=True)
    
    return df