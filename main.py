import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os
import statsmodels.api as sm
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import pickle

# Function to create cache directory if it doesn't exist
def ensure_cache_dir():
    cache_dir = 'data_cache'
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir

# Function to generate cache file paths
def get_cache_path(ticker, start_date, end_date):
    cache_dir = ensure_cache_dir()
    return os.path.join(cache_dir, f"{ticker}_{start_date}_{end_date}.pkl")

# Function to download or load from cache
def get_cached_data(ticker, start_date, end_date=None, force_download=False):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    cache_path = get_cache_path(ticker, start_date, end_date)
    
    # If cache exists and not forcing a download, load from cache
    if os.path.exists(cache_path) and not force_download:
        print(f"Loading {ticker} data from cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    # Otherwise download the data
    print(f"Downloading {ticker} data from Yahoo Finance...")
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    
    # Save to cache
    with open(cache_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Data cached to {cache_path}")
    return data

def download_sp500_data(start_date, end_date=None, force_download=False):
    """
    Download S&P 500 data once with extended date ranges to support all calculations.
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # For forward-looking volatility extend the end date by at least 30 days (21 trading days)
    forward_end_date = (datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=40)).strftime('%Y-%m-%d')
    
    # Get data from at least 100 days before the start date to calculate 3-month volatility and returns
    extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=100)).strftime('%Y-%m-%d')
        
    # Get cached S&P 500 data
    sp500 = get_cached_data('^GSPC', extended_start, forward_end_date, force_download)
    
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

def get_vix_data(start_date, end_date=None, force_download=False):
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
        
    # Extend the end date by one day to ensure there is for the last day in the range
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    extended_end = (end_date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Get VIX data (ticker: ^VIX)
    vix_data = get_cached_data('^VIX', start_date, extended_end, force_download)
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


def ols_time_split(
        df,
        target="forward_volatility_1m",
        date_col="Date",
        n_splits=5,
        test_size=250,
        add_const=True):
    """
    Ordinary Least Squares with expanding-window, time-aware CV.

    Parameters
    ----------
    df         : DataFrame containing features, target, and date column.
    target     : Dependent variable column name.
    date_col   : Date column used to keep chronological order.
    n_splits   : Number of walk-forward folds.
    test_size  : Rows in each validation window (≈ trading days).
    add_const  : If True, include an intercept term.

    Returns
    -------
    cv_scores  : DataFrame with per-fold and average RMSE / R².
    final_res  : statsmodels OLS results on the full sample.
    """

    df = (df.dropna(subset=[target])
            .sort_values(date_col)
            .reset_index(drop=True))

    y = df[target].astype(float)
    X = df.drop(columns=[target, date_col]).astype(float)
    if add_const:
        X = sm.add_constant(X)

    # walk-forward CV
    tscv   = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    rows   = []

    for i, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        res   = sm.OLS(y.iloc[train_idx], X.iloc[train_idx]).fit()
        y_hat = res.predict(X.iloc[test_idx])

        rmse = mean_squared_error(y.iloc[test_idx], y_hat, squared=False)
        r2   = 1 - ((y.iloc[test_idx] - y_hat)**2).sum() / \
                   ((y.iloc[test_idx] - y.iloc[test_idx].mean())**2).sum()

        rows.append({"fold": i, "RMSE": rmse, "R2": r2})
        print(f"Fold {i}: RMSE={rmse:.4f}  R²={r2:.3f}")

    # summary row
    avg_rmse = np.mean([r["RMSE"] for r in rows])
    avg_r2   = np.mean([r["R2"]  for r in rows])
    rows.append({"fold": "Average", "RMSE": avg_rmse, "R2": avg_r2})

    cv_scores = pd.DataFrame(rows).set_index("fold")

    # final full-sample fit
    final_res = sm.OLS(y, X).fit()

    print("\n=== Cross-validated performance ===")
    print(cv_scores)
    print("\n=== Final OLS coefficients (full sample) ===")
    print(final_res.params)

    return cv_scores, final_res

def evaluate_model_metrics(data, n_splits=5):
    """
    Evaluate model using time series cross-validation, reporting only RMSE and MAE metrics.
    
    Parameters:
    - data: DataFrame containing features and target variable
    - n_splits: Number of splits for time series cross-validation
    
    Returns:
    - Results DataFrame
    """
    print("\nEvaluating model metrics with time series cross-validation...")
    
    # Prepare data
    if 'Date' in data.columns:
        # Ensure data is sorted by date
        data = data.sort_values('Date')
        X = data.drop(['Date', 'forward_volatility_1m'], axis=1)
        dates = data['Date']
    else:
        X = data.drop('forward_volatility_1m', axis=1)
        dates = range(len(X))
    
    y = data['forward_volatility_1m']
    
    # Use TimeSeriesSplit for time-ordered cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Get date ranges for reporting
        if 'Date' in data.columns:
            train_start = dates.iloc[train_idx].min()
            train_end = dates.iloc[train_idx].max()
            test_start = dates.iloc[test_idx].min()
            test_end = dates.iloc[test_idx].max()
        else:
            train_start = train_idx[0]
            train_end = train_idx[-1]
            test_start = test_idx[0]
            test_end = test_idx[-1]
        
        print(f"\nFold {fold+1}")
        print(f"Train: {train_start} to {train_end} ({len(X_train)} samples)")
        print(f"Test: {test_start} to {test_end} ({len(X_test)} samples)")
        
        # Fit model (OLS linear regression)
        model = sm.OLS(y_train, sm.add_constant(X_train)).fit()
        
        # Predict with model
        predictions = model.predict(sm.add_constant(X_test))
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        # Store results
        results.append({
            'fold': fold+1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'rmse': rmse,
            'mae': mae
        })
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Print average results
    avg_rmse = results_df['rmse'].mean()
    avg_mae = results_df['mae'].mean()
    
    print("\n=== Average Results ===")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    
    # Create visualization
    plt.figure(figsize=(14, 6))
    
    # Plot metrics as a bar graph
    bar_width = 0.35
    x = results_df['fold']
    x_pos = np.arange(len(x))
    
    # Create bars
    plt.bar(x_pos - bar_width/2, results_df['rmse'], bar_width, label='RMSE', color='blue', alpha=0.7)
    plt.bar(x_pos + bar_width/2, results_df['mae'], bar_width, label='MAE', color='red', alpha=0.7)
    
    # Add fold numbers to x-axis
    plt.xticks(x_pos, results_df['fold'])
    
    # Add value labels on top of bars
    for i, (rmse, mae) in enumerate(zip(results_df['rmse'], results_df['mae'])):
        plt.text(i - bar_width/2, rmse + 0.005, f'{rmse:.4f}', ha='center', va='bottom')
        plt.text(i + bar_width/2, mae + 0.005, f'{mae:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Fold')
    plt.ylabel('Error')
    plt.title('Model Error Metrics by Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save visualization
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig('visualizations/OLS_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to 'visualizations/OLS_performance.png'")
    
    return results_df

if __name__ == "__main__":
    start_date = '2000-01-01'
    end_date = '2024-12-31'
    
    # Set to True to force new download instead of using cache
    force_download = False
    
    # Download S&P 500 data
    sp500_data = download_sp500_data(start_date, end_date, force_download)
    
    # Get feature data
    sp500_features = get_sp500_data(sp500_data, start_date, end_date)
    vix_data = get_vix_data(start_date, end_date, force_download)
    
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
    #data_exploration(combined_data)
    
    # Drop rows with NaN values before evaluation
    combined_data_clean = combined_data.dropna()
    
    # Evaluate model metrics
    evaluation_results = evaluate_model_metrics(combined_data_clean)
