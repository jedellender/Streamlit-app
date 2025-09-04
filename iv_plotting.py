import plotly.graph_objects as go
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import streamlit as st

def create_vol_surface_from_real_data(filtered_df, selected_ticker, current_prices):
    """
    Create 3D volatility surface using your real options data
    """
    if filtered_df.empty or len(filtered_df) < 10:
        st.warning(f"Insufficient options data for {selected_ticker}. Need at least 10 data points.")
        return None, None
    
    # Get current spot price
    spot_price = current_prices.get(selected_ticker, 100)
    
    # Calculate days to expiry if not already present
    if 'days_to_expiry' not in filtered_df.columns:
        df_copy = filtered_df.copy()
        df_copy['days_to_expiry'] = (pd.to_datetime(df_copy['expirationDate']) - pd.Timestamp.now()).dt.days
        # Filter out expired options
        df_copy = df_copy[df_copy['days_to_expiry'] > 0]
    else:
        df_copy = filtered_df[filtered_df['days_to_expiry'] > 0].copy()
    
    if df_copy.empty:
        st.warning("No options with positive days to expiry found.")
        return None, None
    
    # Get the data we need
    strikes = df_copy['strike'].values
    days = df_copy['days_to_expiry'].values
    implied_vols = df_copy['impliedVolatility'].values
    volumes = df_copy['volume'].values if 'volume' in df_copy.columns else np.ones(len(strikes))
    
    # Remove any NaN values
    valid_mask = ~(np.isnan(strikes) | np.isnan(days) | np.isnan(implied_vols))
    strikes = strikes[valid_mask]
    days = days[valid_mask]
    implied_vols = implied_vols[valid_mask]
    volumes = volumes[valid_mask]
    
    if len(strikes) < 10:
        st.warning("Insufficient valid data points after cleaning.")
        return None, None
    
    try:
        # Create interpolated surface for smooth visualisation
        strike_range = np.linspace(strikes.min(), strikes.max(), 40)
        days_range = np.linspace(max(1, days.min()), days.max(), 40)
        X, Y = np.meshgrid(strike_range, days_range)
        
        # Interpolate implied volatilities onto the mesh
        Z = griddata(
            (strikes, days), 
            implied_vols, 
            (X, Y), 
            method='linear',  # Use linear for more robust interpolation with real data
            fill_value=np.nan
        )
        
        # Create the 3D surface plot
        fig = go.Figure()
        
        # Add main surface
        fig.add_trace(go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale='Viridis',
            name='Volatility Surface',
            colorbar=dict(
                title='Implied Volatility',
                tickformat='.1%',
                len=0.9,          # Length of colorbar
                x=1.2,           
            ),
            hovertemplate='<b>Strike:</b> $%{x:.2f}<br>' +
                          '<b>Days to Expiry:</b> %{y:.0f}<br>' +
                          '<b>Implied Vol:</b> %{z:.2%}<br>' +
                          '<extra></extra>',
            opacity=0.8
        ))
        
        # Add scatter points for actual market data - size by volume
        # Normalise volume for marker size
        max_volume = max(volumes) if max(volumes) > 0 else 1
        marker_sizes = 3 + 15 * (volumes / max_volume)  # Size between 3-18
        
        fig.add_trace(go.Scatter3d(
            x=strikes,
            y=days,
            z=implied_vols,
            mode='markers',
            marker=dict(
                size=marker_sizes,
                color=volumes,
                colorscale='Reds',
                opacity=0.8,
                colorbar=dict(
                    title="Volume",
                    x=1.1
                )
            ),
            name='Market Data',
            hovertemplate='<b>Strike:</b> $%{x:.2f}<br>' +
                          '<b>Days to Expiry:</b> %{y:.0f}<br>' +
                          '<b>Implied Vol:</b> %{z:.2%}<br>' +
                          '<b>Volume:</b> %{marker.color:.0f}<br>' +
                          '<extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title={
                'text': f'{selected_ticker} Volatility Surface<br><sub>Spot: ${spot_price:.2f} | Data Points: {len(strikes)}</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            scene=dict(
                xaxis_title='Strike Price ($)',
                yaxis_title='Days to Expiry',
                zaxis_title='Implied Volatility',
                xaxis=dict(tickformat='$.0f'),
                yaxis=dict(tickformat='.0f'),
                zaxis=dict(tickformat='.1%'),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectratio=dict(x=1, y=1, z=1)
            ),
            height=600,
            margin=dict(l=0, r=0, t=80, b=0),
            showlegend=True
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
        st.error(f"Error creating volatility surface: {str(e)}")
        return None, None

def display_vol_surface_metrics(summary_df, spot_price):
    """
    Display key volatility surface metrics
    """
    if summary_df is None or summary_df.empty:
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "ATM Vol", 
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