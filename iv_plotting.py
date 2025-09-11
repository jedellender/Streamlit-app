import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import streamlit as st
from scipy.ndimage import gaussian_filter1d, gaussian_filter
from scipy.interpolate import interp1d
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
            mode='lines', name='IV Smile',
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
    
    return fig