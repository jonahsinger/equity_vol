import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_sp500_data(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Get data from at least 90 days (3 months) before the start date to calculate volatility
    extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=100)).strftime('%Y-%m-%d')
        
    # Get S&P 500 data using the ^GSPC ticker
    sp500 = yf.download('^GSPC', start=extended_start, end=end_date, auto_adjust=True)
    
    # Calculate daily returns
    sp500['daily_return'] = sp500['Close'].pct_change()
    
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
    sp500 = sp500[sp500['Date'] >= start_date]
    
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
        
    # Get VIX data (ticker: ^VIX)
    vix_data = yf.download('^VIX', start=start_date, end=end_date, auto_adjust=True)
    vix_data = vix_data.reset_index()
    
    # Only keep the Date and Close columns
    vix_df = pd.DataFrame()
    vix_df['Date'] = vix_data['Date']
    vix_df['VIX'] = vix_data['Close']  # Rename Close to VIX for clarity
    
    return vix_df

if __name__ == "__main__":
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    # Get S&P 500 data and VIX data separately
    sp500_data = get_sp500_data(start_date, end_date)
    vix_data = get_vix_data(start_date, end_date)

    # Merge the two dataframes on Date
    combined_data = pd.merge(sp500_data, vix_data, on='Date', how='left')
    
    # Display the first few rows with all columns
    print(combined_data.head())
    
    # Count total missing values
    total_missing = combined_data.isna().sum().sum()
    
    if total_missing == 0:
        print("No missing values found in the dataset.")
