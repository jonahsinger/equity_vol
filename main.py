import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_sp500_data(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Get S&P 500 data using the ^GSPC ticker
    sp500 = yf.download('^GSPC', start=start_date, end=end_date, auto_adjust=True)
    sp500 = sp500.reset_index()
    
    clean_df = pd.DataFrame()
    clean_df['Date'] = sp500['Date']
    clean_df['Close'] = sp500['Close']
    clean_df['Volume'] = sp500['Volume']
    
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
    
    print(combined_data.head())
