from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_3d_copula(filtered_df, value_col='impliedVolatility', option_type='Call'):

    # Additional option type filtering if needed
    if option_type != 'All':
        df = filtered_df[filtered_df['optionType'] == option_type]
    else:
        df = filtered_df
    
    if df.empty or len(df) < 10:
        return None
    
    # Calculate days to expiry if not already present
    if 'days_to_expiry' not in df.columns:
        df = df.copy()
        df['days_to_expiry'] = (pd.to_datetime(df['expirationDate']) - pd.Timestamp.now()).dt.days
    
    # Get data points
    strikes = df['strike'].values
    days = df['days_to_expiry'].values  
    values = df[value_col].values
    
    # Rest of your 3D plotting code...
    # Create meshgrid, interpolate, plot surface
    strike_range = np.linspace(strikes.min(), strikes.max(), 30)
    days_range = np.linspace(days.min(), days.max(), 30)
    X, Y = np.meshgrid(strike_range, days_range)
    
    Z = griddata((strikes, days), values, (X, Y), method='linear')
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Days to Expiry') 
    ax.set_zlabel(value_col)
    ax.set_title(f'3D {value_col} Surface')
    
    plt.colorbar(surf, shrink=0.5, aspect=5)
    return fig