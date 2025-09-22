import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import streamlit as st
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d

def calculate_dual_population_skew(expiry_data, spot_price):
    """Calculate separate IV skew slopes for calls and puts"""
    if len(expiry_data) < 6:  # Need at least 3 calls and 3 puts
        return None, None
    
    results = {}
    
    for option_type in ['Call', 'Put']:
        # Filter for specific option type
        type_data = expiry_data[expiry_data['optionType'] == option_type].copy()
        if len(type_data) < 3:
            results[option_type] = None
            continue
        
        type_data['log_moneyness'] = np.log(type_data['strike'] / spot_price)
        type_data = type_data.sort_values('log_moneyness')
        
        # Different moneyness ranges for calls vs puts
        if option_type == 'Call':
            # Calls: focus on ATM to OTM (positive log moneyness)
            reasonable_range = (type_data['log_moneyness'] >= -0.1) & (type_data['log_moneyness'] <= 0.3)
        else:
            # Puts: focus on OTM to ATM (negative log moneyness)
            reasonable_range = (type_data['log_moneyness'] >= -0.3) & (type_data['log_moneyness'] <= 0.1)
        
        type_data = type_data[reasonable_range]
        
        if len(type_data) < 3:
            results[option_type] = None
            continue
        
        x = type_data['log_moneyness'].values
        y = type_data['impliedVolatility'].values
        
        # Remove any NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x = x[valid_mask]
        y = y[valid_mask]
        
        if len(x) < 3:
            results[option_type] = None
            continue
        
        try:
            # Linear regression slope
            slope = np.polyfit(x, y, 1)[0]
            results[option_type] = slope
        except:
            results[option_type] = None
    
    return results.get('Call'), results.get('Put')

def create_vol_surface(filtered_df, selected_ticker, current_prices, dte_max=210):
    if filtered_df.empty or len(filtered_df) < 10:
        st.warning(f"Insufficient options data for {selected_ticker}. Need at least 10 data points.")
        return None, None
    
    # Get current price
    spot_price = current_prices.at[selected_ticker, 'price']

    # Calculate days to expiry if not already present
    df_copy = filtered_df.copy()
    if 'days_to_expiry' not in filtered_df.columns:
        df_copy['days_to_expiry'] = (pd.to_datetime(df_copy['expirationDate']) - pd.Timestamp.now()).dt.days
    

    df_copy = df_copy[ (df_copy['days_to_expiry'] > 0) & (df_copy['days_to_expiry'] < dte_max) ]
    
    if df_copy.empty:
        st.warning("No options with positive days to expiry found.")
        return None, None
    
    # Get the data we need
    strikes = df_copy['strike'].values
    days = df_copy['days_to_expiry'].values
    implied_vols = df_copy['impliedVolatility'].values
    volumes = df_copy['volume'].values if 'volume' in df_copy.columns else np.ones(len(strikes))
    
    # remove outliers 
    rolling_mean = df_copy['impliedVolatility'].rolling(5, center=True).mean()
    outlier_mask = abs(df_copy['impliedVolatility'] - rolling_mean) > 0.1
    df_copy = df_copy[~outlier_mask]

    # extract arrays
    strikes = df_copy['strike'].values
    days = df_copy['days_to_expiry'].values
    implied_vols = df_copy['impliedVolatility'].values
    volumes = df_copy['volume'].values

    # NaN filtering
    valid_mask = ~(np.isnan(strikes) | np.isnan(days) | np.isnan(implied_vols))
    strikes = strikes[valid_mask]
    days = days[valid_mask]
    implied_vols = implied_vols[valid_mask]
    volumes = volumes[valid_mask]
        

    if len(strikes) < 10:
        st.warning("Insufficient valid data points after cleaning.")
        return None, None
    
    try:
        
        # format variables
        log_moneyness = np.log(spot_price/strikes)
        dte = days / 252.0
        sqrt_dte = np.sqrt(dte)

        # interpolate log moneyness and sqrt DTE
        log_moneyness_range = np.linspace(-1, 1, 50) # max, min, resolution
        sqrt_dte_range = np.linspace(np.sqrt(max(1/252.0, dte.min())), np.sqrt(dte.max()), 50)
        X, Y_sqrt = np.meshgrid(log_moneyness_range, sqrt_dte_range)
        
        # Interpolate IV
        Z = griddata(
            (log_moneyness, sqrt_dte), 
            implied_vols, 
            (X, Y_sqrt ), 
            method='cubic',  
            fill_value=np.nan
        )

        Z_linear = griddata((log_moneyness, sqrt_dte), implied_vols, (X, Y_sqrt), 
                   method='linear', fill_value=np.nan)

        Z_nearest = griddata((log_moneyness, sqrt_dte), implied_vols, (X, Y_sqrt), 
                    method='nearest')

        # Use linear where available, nearest only for small gaps
        Z = np.where(np.isnan(Z_linear), Z_nearest, Z_linear)
        Z = gaussian_filter(Z, sigma=1.5)  # Light smoothing after interpolation
        #print(f"Non-NaN Z values: {np.sum(~np.isnan(Z))}/{Z.size}")

        # reset Y for display (but interpolating in sqrt space) 
        Y = 252 * Y_sqrt **2

        #X = np.exp(X) 

        # Create the 3D surface plot

        fig = go.Figure()

        # Add main surface
        fig.add_trace(go.Surface(
            x=X, # S/K
            y=Y, # DTE
            z=Z, # IV
            colorscale='Viridis',
            name='Volatility Surface',
            colorbar=dict(
                title='Implied Volatility',
                tickformat='.1%',
                len=0.9,          # Length of colorbar
                x=1.2,           
                
            ),
            hovertemplate='<b>Moneyness:</b> %{x:.2f}<br>' +
                          '<b>Days to Expiry:</b> %{y:.0f}<br>' +
                          '<b>Implied Vol:</b> %{z:.2%}<br>' +
                          '<extra></extra>',
            opacity=0.8
        ))
        
        # Add scatter points for actual market data - size by volume
        # Normalise volume for marker size
        max_volume = max(volumes) if max(volumes) > 0 else 1
        marker_sizes = 5 + 15 * (volumes / max_volume)  # Size between 3-18
        
        
        fig.add_trace(go.Scatter3d(
            x=log_moneyness,
            y=days,
            z=implied_vols,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=volumes,
                colorscale='emrld',
                opacity=0.3,
                colorbar=dict(
                    title="Volume",
                    x=0.02
                    
                )
            ),
            
            name='Click to toggle: Data points',
            hovertemplate='<b>Moneyness:</b> $%{x:.2f}<br>' +
                          '<b>Days to Expiry:</b> %{y:.0f}<br>' +
                          '<b>Implied Vol:</b> %{z:.2%}<br>' +
                          '<b>Volume:</b> %{marker.color:.0f}<br>' +
                          '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'Spot: ${spot_price:.2f} | Data Points: {len(log_moneyness)}</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },

        
            scene=dict(
                xaxis_title='Log Moneyness',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility',
                xaxis=dict(tickformat='.2f'),
                yaxis=dict(tickformat='.0f'),
                zaxis=dict(tickformat='.2%'),
                camera=dict(
                eye=dict(x=-2.5, y=-2.5, z=1),  # View from negative y-axis
                up=dict(x=0, y=0, z=1) 
                ),
                aspectratio=dict(x=1.5, y=1.5, z=1)
            ),
            height=500,
            margin=dict(l=20, r=80, t=30, b=20),
            showlegend=True,
            legend=dict(
            x=0.02,  
            y=1.1,
            font=dict(size=16, family="Arial"),
            bgcolor="rgba(255,255,255,0.8)"
    )
        )
        
        
        # Create summary dataframe
        summary_df = pd.DataFrame({
            'Strike': strikes,
            'Days_to_Expiry': days,
            'Implied_Vol': implied_vols,
            'Volume': volumes,
            'Moneyness': strikes / spot_price
        })
        return fig, summary_df
    
    except Exception as e:
        st.error(f"Error creating volatility surface Message: {e}")
        return None, None

def display_vol_surface_metrics(summary_df, current_price):
    if summary_df is None or summary_df.empty:
        return
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Avg ATM Vol", 
            f"{summary_df.loc[abs(summary_df['Moneyness'] - 1.0).idxmin(), 'Implied_Vol']:.1%}"
        )
    with col2:
        st.metric(
            "Vol Range", 
            f"{summary_df['Implied_Vol'].min():.1%} - {summary_df['Implied_Vol'].max():.1%}"
        )
    
    with col3:
        st.metric(
            "Days Range", 
            f"{int(summary_df['Days_to_Expiry'].min())} - {int(summary_df['Days_to_Expiry'].max())}"
        )
    
    with col4:
        st.metric(
            "Strike Range", 
            f"${summary_df['Strike'].min():.0f} - ${summary_df['Strike'].max():.0f}"
        )

def plot_vol_2d(filtered_df, selected_ticker, current_prices):
    fig = go.Figure()
    
    # Get current price for moneyness
    spot_price = current_prices.at[selected_ticker, 'price']
    
    # Get data for nearest expiry calls
    nearest_exp = filtered_df['expirationDate'].min()
    data = filtered_df[
        (filtered_df['expirationDate'] == nearest_exp)&
    (filtered_df['optionType'] == 'Call')
    ].copy()
    
    
    # Calculate moneyness and filter
    data['log_moneyness'] = np.log(data['strike'] / spot_price)
    data = data.sort_values('log_moneyness')
    
    # Apply moneyness and IV filters
    reasonable_range = (data['log_moneyness'] >= -0.2) & (data['log_moneyness'] <= 0.3)
    reasonable_iv = (data['impliedVolatility'] > 0) & (data['impliedVolatility'] < 3)
    reasonables = reasonable_range & reasonable_iv
    data = data[reasonables]
    
    if len(data) < 3:
        x, y = np.array([]), np.array([])
    else:
        x = data['log_moneyness'].values
        y = data['impliedVolatility'].values
    
    # Create smooth interpolation
    if len(x) >= 4:  # Need minimum points for interpolation
        x_smooth = np.linspace(x.min(), x.max(), 50)
        f = interp1d(x, y, kind='linear')
        y_smooth = f(x_smooth)
        y_smooth = gaussian_filter1d(y_smooth, sigma=3)
        
        
        # Plot smooth curve
        fig.add_trace(go.Scatter(
            x=x_smooth, y=y_smooth,
            mode='lines', name='IV Skew',
            line=dict(width=3)
        ))
    
    # Add original data points
    fig.add_trace(go.Scatter(
        x=x, y=y,
        mode='markers', name='Market Data',
        marker=dict(opacity=0.4, size=8, color='red')
    ))
    
    # Add ATM line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="ATM")
    
    days_to_exp = (nearest_exp - pd.Timestamp.now()).days +1
    fig.update_layout(
        title=f'{selected_ticker} Vol Skew - DTE: {days_to_exp}D (Exp: {nearest_exp.strftime("%d %b %Y")})',
        xaxis_title='Log Moneyness',
        yaxis_title='Implied Volatility',
        yaxis_tickformat='.1%'
    )
    
    # Return simplified consensus data without z-scores
    consensus_data = {
        'days_to_exp': days_to_exp,
        'data_points': len(x)
    }
    
    return fig, consensus_data

def plot_dual_population_skew(filtered_df, selected_ticker, current_prices):
    """Plot volatility skew using dual-population approach (separate calls and puts)"""
    fig = go.Figure()
    
    # Get current price for moneyness
    spot_price = current_prices.at[selected_ticker, 'price']
    
    # Get data for nearest expiry
    nearest_exp = filtered_df['expirationDate'].min()
    data = filtered_df[filtered_df['expirationDate'] == nearest_exp].copy()
    
    # Calculate moneyness for all data
    data['log_moneyness'] = np.log(data['strike'] / spot_price)
    
    colors = {'Call': '#1f77b4', 'Put': '#ff7f0e'}
    
    for option_type in ['Call', 'Put']:
        type_data = data[data['optionType'] == option_type].copy()
        
        if len(type_data) < 3:
            continue
            
        # Apply moneyness filters
        if option_type == 'Call':
            reasonable_range = (type_data['log_moneyness'] >= -0.1) & (type_data['log_moneyness'] <= 0.3)
        else:
            reasonable_range = (type_data['log_moneyness'] >= -0.3) & (type_data['log_moneyness'] <= 0.1)
        
        reasonable_iv = (type_data['impliedVolatility'] > 0) & (type_data['impliedVolatility'] < 3)
        reasonables = reasonable_range & reasonable_iv
        type_data = type_data[reasonables]
        
        if len(type_data) < 3:
            continue
            
        type_data = type_data.sort_values('log_moneyness')
        x = type_data['log_moneyness'].values
        y = type_data['impliedVolatility'].values
        
        # Plot data points
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers', name=f'{option_type} Data',
            marker=dict(opacity=0.6, size=8, color=colors[option_type])
        ))
        
        # Add regression line if enough points
        if len(x) >= 3:
            try:
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                
                fig.add_trace(go.Scatter(
                    x=x_line, y=y_line,
                    mode='lines', name=f'{option_type} Fit',
                    line=dict(width=2, color=colors[option_type], dash='dash')
                ))
            except:
                pass
    
    # Add ATM line
    fig.add_vline(x=0, line_dash="dash", line_color="gray", annotation_text="ATM")
    
    days_to_exp = (nearest_exp - pd.Timestamp.now()).days + 1
    
    # Calculate dual population skews for display
    call_slope, put_slope = calculate_dual_population_skew(data, spot_price)
    title_text = f'{selected_ticker} Skew - DTE: {days_to_exp}D'
    
    fig.update_layout(
        title=title_text,
        xaxis_title='Log Moneyness',
        yaxis_title='Implied Volatility',
        yaxis_tickformat='.1%'
    )
    
    return fig, {'call_slope': call_slope, 'put_slope': put_slope, 'data_points': len(data)}