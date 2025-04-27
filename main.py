import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def get_sp500_data(start_date, end_date=None):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    simplified_data = sp500[['Close', 'Volume']]
    simplified_data = simplified_data.reset_index()
    
    return simplified_data

if __name__ == "__main__":
    start_date = '2023-01-01'
    end_date = '2023-12-31'
    
    sp500_data = get_sp500_data(start_date, end_date)
    print(sp500_data.head())