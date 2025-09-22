import pandas as pd
import numpy as np

def get_top_liquid_options(df, top_n=3):
    """
    Find the most liquid options based on volume and open interest
    Returns the top N most liquid options with key metrics
    """
    if df is None or df.empty:
        return None
    
    # Create liquidity score (combine volume and open interest)
    df = df.copy()
    
    # Normalize volume and open interest to 0-1 scale for fair weighting
    if 'volume' in df.columns and 'openInterest' in df.columns:
        max_vol = df['volume'].max() if df['volume'].max() > 0 else 1
        max_oi = df['openInterest'].max() if df['openInterest'].max() > 0 else 1
        
        df['norm_volume'] = df['volume'] / max_vol
        df['norm_oi'] = df['openInterest'] / max_oi
        
        # Liquidity score: 60% volume, 40% open interest (volume more important for immediate tradability)
        df['liquidity_score'] = (df['norm_volume'] * 0.6) + (df['norm_oi'] * 0.4)
    else:
        # Fallback to just volume if open interest not available
        df['liquidity_score'] = df['volume'] if 'volume' in df.columns else 0
    
    # Get top N most liquid options
    top_liquid = df.nlargest(top_n, 'liquidity_score')
    
    # Select relevant columns for display
    display_columns = [
        'optionType', 'strike', 'expirationDate', 'volume', 'openInterest',
        'impliedVolatility', 'lastPrice', 'bid', 'ask', 'moneyness'
    ]
    
    # Only include columns that exist in the dataframe
    available_columns = [col for col in display_columns if col in top_liquid.columns]
    
    return top_liquid[available_columns]

def format_top_liquid_options(top_liquid_df):
    """
    Format the top liquid options for nice display
    Returns formatted dataframe and summary metrics
    """
    if top_liquid_df is None or top_liquid_df.empty:
        return None, None
    
    # Create a formatted copy
    formatted_df = top_liquid_df.copy()
    
    # Format columns for better display
    if 'expirationDate' in formatted_df.columns:
        formatted_df['Expiry'] = pd.to_datetime(formatted_df['expirationDate']).dt.strftime('%m/%d')
    
    if 'impliedVolatility' in formatted_df.columns:
        formatted_df['IV'] = (formatted_df['impliedVolatility'] * 100).round(1).astype(str) + '%'
    
    if 'lastPrice' in formatted_df.columns:
        formatted_df['Price'] = '$' + formatted_df['lastPrice'].round(2).astype(str)
    
    if 'moneyness' in formatted_df.columns:
        formatted_df['Moneyness'] = formatted_df['moneyness'].round(3)
    
    # Create summary metrics
    total_volume = formatted_df['volume'].sum() if 'volume' in formatted_df.columns else 0
    avg_iv = formatted_df['impliedVolatility'].mean() if 'impliedVolatility' in formatted_df.columns else 0
    call_count = len(formatted_df[formatted_df['optionType'] == 'Call']) if 'optionType' in formatted_df.columns else 0
    put_count = len(formatted_df[formatted_df['optionType'] == 'Put']) if 'optionType' in formatted_df.columns else 0
    
    summary_metrics = {
        'total_volume': total_volume,
        'avg_iv': avg_iv,
        'call_count': call_count,
        'put_count': put_count
    }
    
    # Select columns for final display
    display_columns = ['optionType', 'strike', 'Expiry', 'volume', 'openInterest', 'IV', 'Price', 'Moneyness']
    final_columns = [col for col in display_columns if col in formatted_df.columns]
    
    return formatted_df[final_columns], summary_metrics

def get_liquidity_insights(summary_metrics):
    """
    Generate insights from the liquidity analysis
    """
    if summary_metrics is None:
        return "No liquidity data available"
    
    total_vol = summary_metrics['total_volume']
    call_count = summary_metrics['call_count']
    put_count = summary_metrics['put_count']
    
    insights = []
    
    # Volume insight
    if total_vol > 1000:
        insights.append("🔥 HIGH ACTIVITY - Strong trading volume")
    elif total_vol > 100:
        insights.append("📊 MODERATE ACTIVITY - Decent liquidity")
    else:
        insights.append("📉 LOW ACTIVITY - Limited liquidity")
    
    # Call/Put bias
    if call_count > put_count:
        insights.append("📈 CALL BIAS - More liquid call options")
    elif put_count > call_count:
        insights.append("📉 PUT BIAS - More liquid put options")
    else:
        insights.append("⚖️ BALANCED - Equal call/put liquidity")
    
    return " | ".join(insights)