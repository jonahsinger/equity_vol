import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import os

def download_sp500_data(start_date, end_date=None):
    """
    Download S&P 500 data once with extended date ranges to support all calculations.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # For forward-looking volatility extend the end date by at least 30 days (21 trading days)
    forward_end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=40)).strftime('%Y-%m-%d')
    
    # Get data from at least 100 days before the start date to calculate 3-month volatility and returns
    extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=100)).strftime('%Y-%m-%d')
        
    # Get S&P 500 data using the ^GSPC ticker, with extended date range
    print("Downloading S&P 500 data...")
    sp500 = yf.download('^GSPC', start=extended_start, end=forward_end_date, auto_adjust=True)
    
    # Calculate daily returns
    sp500['daily_return'] = sp500['Close'].pct_change()
    
    return sp500

def get_sp500_data(sp500_data, start_date, end_date=None):
    """
    Process the already downloaded S&P 500 data to calculate features.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Make a copy to avoid modifying the original
    sp500 = sp500_data.copy()
    
    # Calculate rolling volatilities (annualized)
    # 5 trading days ≈ 1 week, 63 trading days ≈ 3 months, sqrt(252) for annualization
    sp500['volatility_1w'] = sp500['daily_return'].rolling(window=5).std() * np.sqrt(252)
    sp500['volatility_3m'] = sp500['daily_return'].rolling(window=63).std() * np.sqrt(252)
    
    # Calculate returns for different periods
    # 1-week return (5 trading days)
    sp500['return_1w'] = sp500['Close'].pct_change(periods=5)
    
    # 3-month return (63 trading days)
    sp500['return_3m'] = sp500['Close'].pct_change(periods=63)
    
    # Calculate volume changes
    # 1-week volume change
    sp500['volume_change_1w'] = sp500['Volume'].pct_change(periods=5)
    
    # 3-month volume change (63 trading days instead of 21)
    sp500['volume_change_3m'] = sp500['Volume'].pct_change(periods=63)
    
    # Reset the index after calculations
    sp500 = sp500.reset_index()
    
    # Filter to the original date range
    sp500 = sp500[(sp500['Date'] >= start_date) & (sp500['Date'] <= end_date)]
    
    # Create a clean DataFrame with columns in order
    clean_df = pd.DataFrame()
    clean_df['Date'] = sp500['Date']
    clean_df['return_1w'] = sp500['return_1w']
    clean_df['return_3m'] = sp500['return_3m']
    clean_df['volume_change_1w'] = sp500['volume_change_1w']
    clean_df['volume_change_3m'] = sp500['volume_change_3m']
    clean_df['volatility_1w'] = sp500['volatility_1w']
    clean_df['volatility_3m'] = sp500['volatility_3m']
    return clean_df

def get_vix_data(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Extend the end date by one day to ensure there is for the last day in the range
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    extended_end = (end_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Get VIX data (ticker: ^VIX)
    print("Downloading VIX data...")
    vix_data = yf.download('^VIX', start=start_date, end=extended_end, auto_adjust=True)
    vix_data = vix_data.reset_index()
    
    # Only keep the Date and Close columns
    vix_df = pd.DataFrame()
    vix_df['Date'] = vix_data['Date']
    vix_df['VIX'] = vix_data['Close']  # Rename Close to VIX for clarity
    
    return vix_df

def get_forward_volatility(sp500_data, start_date, end_date=None):
    """
    Calculate forward-looking 1-month realized volatility from the already downloaded S&P 500 data.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Make a copy to avoid modifying the original
    sp500 = sp500_data.copy()
    
    # Calculate forward-looking 1-month (21 trading days) realized volatility
    rolled_std = sp500['daily_return'].rolling(window=21).std().shift(-21)
    sp500['forward_volatility_1m'] = rolled_std * np.sqrt(252)  # Annualize
    
    # Reset the index and filter to the original date range
    sp500 = sp500.reset_index()
    sp500 = sp500[(sp500['Date'] >= start_date) & (sp500['Date'] <= end_date)]
    
    # Create a DataFrame with just the Date and forward volatility
    forward_vol_df = pd.DataFrame()
    forward_vol_df['Date'] = sp500['Date']
    forward_vol_df['forward_volatility_1m'] = sp500['forward_volatility_1m']
    
    return forward_vol_df

def data_exploration(data, output_dir='visualizations'):
    """
    Create line graphs and histograms for all features in the dataset
    
    Parameters:
    - data: DataFrame containing all features
    - output_dir: Directory to save visualization files
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 12
    
    # Get all features except Date
    features = [col for col in data.columns if col != 'Date']
    
    # Create line plots for all features
    print(f"Creating line plots for {len(features)} features...")
    for feature in features:
        plt.figure()
        
        plt.plot(data['Date'], data[feature], linewidth=2, color='darkblue')
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        plt.xticks(rotation=90)
        plt.title(f'Time Series: {feature}', fontsize=16)
        plt.ylabel(feature, fontsize=14)
        plt.xlabel('Date', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature}_line.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create histograms for all features
    print(f"Creating histograms for {len(features)} features...")
    for feature in features:
        plt.figure()
        plt.hist(data[feature].dropna(), bins=30, color='darkblue', alpha=0.7)
        
        # Add vertical line for mean and median
        mean_val = data[feature].mean()
        median_val = data[feature].median()
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.4f}')
        plt.axvline(median_val, color='green', linestyle=':', label=f'Median: {median_val:.4f}')
        
        # Set labels and title
        plt.title(f'Distribution: {feature}', fontsize=16)
        plt.xlabel(feature, fontsize=14)
        plt.ylabel('Frequency', fontsize=14)
        plt.legend()
        
        # Add text with descriptive statistics
        stats_text = (f"Mean: {mean_val:.4f}\n"
                     f"Median: {median_val:.4f}\n"
                     f"Std Dev: {data[feature].std():.4f}\n"
                     f"Min: {data[feature].min():.4f}\n"
                     f"Max: {data[feature].max():.4f}")
        
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=12, ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{feature}_hist.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create a correlation matrix
    print("Creating correlation matrix...")
    plt.figure(figsize=(12, 10))
    
    # Calculate correlation matrix
    corr = data[features].corr()
    
    # Create heatmap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                square=True, linewidths=.5, annot=True, fmt=".2f", cbar_kws={"shrink": .8})
    
    plt.title('Correlation Matrix of Features', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"All visualizations saved to '{output_dir}/' directory.")
    
    # Return the correlation with the target variable
    target_correlations = corr['forward_volatility_1m'].sort_values(ascending=False)
    print("\nFeature Correlations with Target (forward_volatility_1m):")
    for feature, correlation in target_correlations.items():
        if feature != 'forward_volatility_1m':
            print(f"{feature}: {correlation:.4f}")
    
    return target_correlations

if __name__ == "__main__":
    start_date = '2000-01-01'
    end_date = '2024-12-31'
    
    # Download S&P 500 data once
    sp500_data = download_sp500_data(start_date, end_date)
    
    # Get feature data
    sp500_features = get_sp500_data(sp500_data, start_date, end_date)
    vix_data = get_vix_data(start_date, end_date)
    
    # Get forward volatility (target variable)
    forward_vol_data = get_forward_volatility(sp500_data, start_date, end_date)

    # Merge the dataframes, ensuring forward_volatility_1m is the last column
    combined_data = pd.merge(sp500_features, vix_data, on='Date', how='left')
    combined_data = pd.merge(combined_data, forward_vol_data, on='Date', how='left')
    
    # Display the first few rows with all columns
    print(combined_data.head())
    
    # Check for missing values
    total_missing = combined_data.isna().sum().sum()
    if total_missing > 0:
        print(f"\nTotal missing values before filling: {total_missing}")
        print("Missing values by column:")
        print(combined_data.isna().sum())
    else:
        print("\nNo missing values found in the dataset.")
    
    # Run data exploration to create visualizations
    data_exploration(combined_data)
