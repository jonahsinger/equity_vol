import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_sp500_data(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    # Get S&P 500 data using the ^GSPC ticker
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    sp500 = sp500.reset_index()
    clean_df = pd.DataFrame()
    clean_df['Date'] = sp500['Date']
    clean_df['Close'] = sp500['Close']
    clean_df['Volume'] = sp500['Volume']
    
    return clean_df

if __name__ == "__main__":
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    sp500_data = get_sp500_data(start_date, end_date)
    print(sp500_data.head()) 