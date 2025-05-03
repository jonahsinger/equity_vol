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
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error
import pickle
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import warnings
warnings.filterwarnings('ignore')

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
    vix_df['VIX'] = vix_data['Close'] / 100  # Divide by 100 to scale to similar range as other features
    
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

def evaluate_model_metrics(data, n_splits=5, verbose=False):
    """
    Evaluate model using time series cross-validation, reporting only RMSE and MAE metrics.
    
    Parameters:
    - data: DataFrame containing features and target variable
    - n_splits: Number of splits for time series cross-validation
    - verbose: If True, print results for each fold
    
    Returns:
    - Results DataFrame
    """
    print("\nEvaluating OLS Linear Regression...")
    
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
    all_coefficients = []
    
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
        
        if verbose:
            print(f"\nFold {fold+1}")
            print(f"Train: {train_start} to {train_end} ({len(X_train)} samples)")
            print(f"Test: {test_start} to {test_end} ({len(X_test)} samples)")
        
        # Fit model (OLS linear regression)
        X_train_const = sm.add_constant(X_train)
        model = sm.OLS(y_train, X_train_const).fit()
        
        # Store coefficients
        coef_dict = {'const': model.params['const']}
        for feature, coef in zip(X_train.columns, model.params[1:]):
            coef_dict[feature] = coef
        all_coefficients.append(coef_dict)
        
        # Predict with model
        predictions = model.predict(sm.add_constant(X_test))
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        if verbose:
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
    
    print("\n=== OLS Linear Regression - Average Results ===")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    
    # Calculate and display average coefficients
    print("\n=== Average Model Weights ===")
    avg_coefs = {}
    for coef_dict in all_coefficients:
        for feature, value in coef_dict.items():
            if feature not in avg_coefs:
                avg_coefs[feature] = []
            avg_coefs[feature].append(value)
    
    avg_coef_df = pd.DataFrame({
        'Feature': list(avg_coefs.keys()),
        'Average Weight': [np.mean(values) for values in avg_coefs.values()],
        'Std Dev': [np.std(values) for values in avg_coefs.values()]
    })
    
    # Sort by absolute coefficient value
    avg_coef_df['Abs Weight'] = avg_coef_df['Average Weight'].abs()
    avg_coef_df = avg_coef_df.sort_values('Abs Weight', ascending=False).drop('Abs Weight', axis=1)
    
    print(avg_coef_df.to_string(index=False, float_format='%.6f'))

    # Create visualization of the OLS results but don't print outputs
    create_model_visualizations(results_df, avg_coef_df, "Linear_Least_Squares")
    
    print(f"Visualizations saved to 'visualizations/' directory")
    
    return results_df, avg_coef_df

def create_model_visualizations(results_df, avg_coef_df, model_name):
    """
    Create visualizations for model results.
    
    Parameters:
    - results_df: DataFrame with fold results
    - avg_coef_df: DataFrame with average coefficients
    - model_name: Name of the model for file names
    """
    # Create visualization
    plt.figure(figsize=(14, 6))
    
    # Plot metrics
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
    plt.ylabel('Error Value (RMSE / MAE)')
    plt.title(f'{model_name} Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save visualization
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig(f'visualizations/{model_name}_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a bar chart for feature weights
    plt.figure(figsize=(14, 8))
    features = avg_coef_df['Feature'].tolist()
    weights = avg_coef_df['Average Weight'].tolist()
    errors = avg_coef_df['Std Dev'].tolist()
    
    # Sort by absolute weight for the chart
    sorted_indices = np.argsort(np.abs(weights))[::-1]
    features = [features[i] for i in sorted_indices]
    weights = [weights[i] for i in sorted_indices]
    errors = [errors[i] for i in sorted_indices]
    
    # Create bars with error bars
    bars = plt.bar(range(len(weights)), weights, yerr=errors, capsize=5)
    
    # Color positive and negative bars differently
    for i, bar in enumerate(bars):
        if weights[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')
    
    # Add value labels on top/bottom of bars
    for i, bar in enumerate(bars):
        height = weights[i]
        plt.text(i, height + (0.01 if height >= 0 else -0.01), 
                 f'{height:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.ylabel('Average Weight')
    plt.title(f'Average Feature Weights from {model_name}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(f'visualizations/{model_name}_weights.png', dpi=300, bbox_inches='tight')
    plt.close()

def ridge_regression_cv(data, n_splits=5, alphas=None, verbose=False):
    """
    Perform Ridge Regression with time series cross-validation to select optimal alpha.
    
    Parameters:
    - data: DataFrame containing features and target variable
    - n_splits: Number of splits for time series cross-validation
    - alphas: List of alpha values to try. If None, a default range will be used.
    - verbose: If True, print results for each fold
    
    Returns:
    - Results DataFrame and optimized coefficients
    """
    print("\nPerforming Ridge Regression with cross-validation...")
    
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
    
    # Default alpha range if not provided
    if alphas is None:
        # Wider range from 0.0001 to 100 with more points
        alphas = np.logspace(-4, 2, 30)
    
    # Use TimeSeriesSplit for time-ordered cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    all_coefficients = []
    best_alphas = []
    
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
        
        if verbose:
            print(f"\nFold {fold+1}")
            print(f"Train: {train_start} to {train_end} ({len(X_train)} samples)")
            print(f"Test: {test_start} to {test_end} ({len(X_test)} samples)")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use GridSearchCV to find optimal alpha within this fold
        ridge_cv = GridSearchCV(
            Ridge(random_state=42),
            {'alpha': alphas},
            cv=5,  # 5-fold CV within the training data
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        ridge_cv.fit(X_train_scaled, y_train)
        best_alpha = ridge_cv.best_params_['alpha']
        best_alphas.append(best_alpha)
        
        if verbose:
            print(f"Optimal alpha for fold {fold+1}: {best_alpha:.6f}")
        
        # Train Ridge with best alpha
        ridge = Ridge(alpha=best_alpha, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        
        # Store coefficients with feature names
        coef_dict = {}
        for feature, coef in zip(X_train.columns, ridge.coef_):
            coef_dict[feature] = coef
        all_coefficients.append(coef_dict)
        
        # Predict on test set
        predictions = ridge.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        if verbose:
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        # Store results
        results.append({
            'fold': fold+1,
            'alpha': best_alpha,
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
    avg_alpha = results_df['alpha'].mean()
    
    print("\n=== Ridge Regression - Average Results ===")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average optimal alpha: {avg_alpha:.6f}")
    
    # Calculate and display average coefficients
    print("\n=== Average Ridge Model Weights ===")
    avg_coefs = {}
    for coef_dict in all_coefficients:
        for feature, value in coef_dict.items():
            if feature not in avg_coefs:
                avg_coefs[feature] = []
            avg_coefs[feature].append(value)
    
    avg_coef_df = pd.DataFrame({
        'Feature': list(avg_coefs.keys()),
        'Average Weight': [np.mean(values) for values in avg_coefs.values()],
        'Std Dev': [np.std(values) for values in avg_coefs.values()]
    })
    
    # Sort by absolute coefficient value
    avg_coef_df['Abs Weight'] = avg_coef_df['Average Weight'].abs()
    avg_coef_df = avg_coef_df.sort_values('Abs Weight', ascending=False).drop('Abs Weight', axis=1)
    
    print(avg_coef_df.to_string(index=False, float_format='%.6f'))
    
    # Create visualizations but don't print outputs
    create_model_visualizations(results_df, avg_coef_df, "Ridge_Regression")
    
    # Plot alpha values by fold
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_splits + 1), best_alphas, color='purple', alpha=0.7)
    plt.axhline(y=avg_alpha, color='red', linestyle='--', label=f'Average: {avg_alpha:.6f}')
    plt.xlabel('Fold')
    plt.ylabel('Optimal Alpha')
    plt.title('Optimal Regularization Parameter (Alpha) by Fold')
    plt.xticks(range(1, n_splits + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/Ridge_Regression_alphas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ridge regression visualizations saved to 'visualizations/' directory")
    
    return results_df, avg_coef_df, best_alphas

def lasso_regression_cv(data, n_splits=5, alphas=None, verbose=False):
    """
    Perform Lasso Regression with time series cross-validation to select optimal alpha.
    
    Parameters:
    - data: DataFrame containing features and target variable
    - n_splits: Number of splits for time series cross-validation
    - alphas: List of alpha values to try. If None, a default range will be used.
    - verbose: If True, print results for each fold
    
    Returns:
    - Results DataFrame and optimized coefficients
    """
    print("\nPerforming Lasso Regression with cross-validation...")
    
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
    
    # Default alpha range if not provided
    if alphas is None:
        # Lasso typically needs smaller alpha values than Ridge
        alphas = np.logspace(-6, 0, 30)
    
    # Use TimeSeriesSplit for time-ordered cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    all_coefficients = []
    best_alphas = []
    
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
        
        if verbose:
            print(f"\nFold {fold+1}")
            print(f"Train: {train_start} to {train_end} ({len(X_train)} samples)")
            print(f"Test: {test_start} to {test_end} ({len(X_test)} samples)")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use GridSearchCV to find optimal alpha within this fold
        lasso_cv = GridSearchCV(
            Lasso(random_state=42, max_iter=10000),
            {'alpha': alphas},
            cv=5,  # 5-fold CV within the training data
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        lasso_cv.fit(X_train_scaled, y_train)
        best_alpha = lasso_cv.best_params_['alpha']
        best_alphas.append(best_alpha)
        
        if verbose:
            print(f"Optimal alpha for fold {fold+1}: {best_alpha:.6f}")
        
        # Train Lasso with best alpha
        lasso = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        
        # Store coefficients with feature names
        coef_dict = {}
        for feature, coef in zip(X_train.columns, lasso.coef_):
            coef_dict[feature] = coef
        all_coefficients.append(coef_dict)
        
        # Predict on test set
        predictions = lasso.predict(X_test_scaled)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        if verbose:
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        # Store results
        results.append({
            'fold': fold+1,
            'alpha': best_alpha,
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
    avg_alpha = results_df['alpha'].mean()
    
    print("\n=== Lasso Regression - Average Results ===")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average optimal alpha: {avg_alpha:.6f}")
    
    # Calculate and display average coefficients
    print("\n=== Average Lasso Model Weights ===")
    avg_coefs = {}
    for coef_dict in all_coefficients:
        for feature, value in coef_dict.items():
            if feature not in avg_coefs:
                avg_coefs[feature] = []
            avg_coefs[feature].append(value)
    
    avg_coef_df = pd.DataFrame({
        'Feature': list(avg_coefs.keys()),
        'Average Weight': [np.mean(values) for values in avg_coefs.values()],
        'Std Dev': [np.std(values) for values in avg_coefs.values()],
        'Zero Count': [sum(1 for v in values if v == 0) for values in avg_coefs.values()]
    })
    
    # Sort by absolute coefficient value
    avg_coef_df['Abs Weight'] = avg_coef_df['Average Weight'].abs()
    avg_coef_df = avg_coef_df.sort_values('Abs Weight', ascending=False).drop('Abs Weight', axis=1)
    
    print(avg_coef_df.to_string(index=False, float_format='%.6f'))
    
    # Create visualizations but don't print outputs
    create_model_visualizations(results_df, avg_coef_df, "Lasso_Regression")
    
    # Plot alpha values by fold
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, n_splits + 1), best_alphas, color='purple', alpha=0.7)
    plt.axhline(y=avg_alpha, color='red', linestyle='--', label=f'Average: {avg_alpha:.6f}')
    plt.xlabel('Fold')
    plt.ylabel('Optimal Alpha')
    plt.title('Optimal Regularization Parameter (Alpha) by Fold')
    plt.xticks(range(1, n_splits + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/Lasso_Regression_alphas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a special bar chart for Lasso feature weights with zero counts
    plt.figure(figsize=(14, 8))
    features = avg_coef_df['Feature'].tolist()
    weights = avg_coef_df['Average Weight'].tolist()
    errors = avg_coef_df['Std Dev'].tolist()
    
    # Sort by absolute weight for the chart
    sorted_indices = np.argsort(np.abs(weights))[::-1]
    features = [features[i] for i in sorted_indices]
    weights = [weights[i] for i in sorted_indices]
    errors = [errors[i] for i in sorted_indices]
    zero_counts = [avg_coef_df.loc[idx, 'Zero Count'] for idx in sorted_indices]
    
    # Create bars with error bars
    bars = plt.bar(range(len(weights)), weights, yerr=errors, capsize=5)
    
    # Color positive and negative bars differently
    for i, bar in enumerate(bars):
        if weights[i] < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')
        
        # Highlight bars with zero coefficients
        if zero_counts[i] > 0:
            bar.set_alpha(0.5)
    
    # Add value labels on top/bottom of bars
    for i, bar in enumerate(bars):
        height = weights[i]
        if abs(height) >= 0.001:  # Only label non-zero coefficients
            plt.text(i, height + (0.01 if height >= 0 else -0.01), 
                     f'{height:.4f}', ha='center', va='bottom' if height >= 0 else 'top')
        if zero_counts[i] > 0:
            plt.text(i, 0, f'0 in {zero_counts[i]}/{n_splits}', ha='center', va='bottom', 
                     rotation=90, fontsize=8, color='gray')
    
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.ylabel('Average Weight')
    plt.title('Average Feature Weights from Lasso Regression')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig('visualizations/Lasso_Regression_weights.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Lasso regression visualizations saved to 'visualizations/' directory")
    
    return results_df, avg_coef_df, best_alphas

def knn_regression_cv(data, n_splits=5, k_values=None, verbose=False):
    """
    Perform K-Nearest Neighbors Regression with time series cross-validation to select optimal k.
    
    Parameters:
    - data: DataFrame containing features and target variable
    - n_splits: Number of splits for time series cross-validation
    - k_values: List of k values to try. If None, a default range will be used.
    - verbose: If True, print results for each fold
    
    Returns:
    - Results DataFrame and optimized k values
    """
    print("\nPerforming KNN Regression with cross-validation...")
    
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
    
    # Default k range if not provided
    if k_values is None:
        # Try a range of k values, considering both small and larger neighborhood sizes
        k_values = list(range(1, 21)) + list(range(25, 101, 5))
    
    # Use TimeSeriesSplit for time-ordered cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    best_k_values = []
    
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
        
        if verbose:
            print(f"\nFold {fold+1}")
            print(f"Train: {train_start} to {train_end} ({len(X_train)} samples)")
            print(f"Test: {test_start} to {test_end} ({len(X_test)} samples)")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Use GridSearchCV to find optimal k within this fold
        knn_cv = GridSearchCV(
            KNeighborsRegressor(weights='distance'),  # Use distance-weighted predictions
            {'n_neighbors': k_values},
            cv=5,  # 5-fold CV within the training data
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        knn_cv.fit(X_train_scaled, y_train)
        best_k = knn_cv.best_params_['n_neighbors']
        best_k_values.append(best_k)
        
        if verbose:
            print(f"Optimal k for fold {fold+1}: {best_k}")
        
        # Train KNN with best k
        knn = KNeighborsRegressor(n_neighbors=best_k, weights='distance')
        knn.fit(X_train_scaled, y_train)
        
        # Predict on test set
        predictions = knn.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        if verbose:
            print(f"RMSE with k={best_k}: {rmse:.4f}")
            print(f"MAE with k={best_k}: {mae:.4f}")
        
        # Store results
        results.append({
            'fold': fold+1,
            'k': best_k,
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
    avg_k = results_df['k'].mean()
    
    print("\n=== KNN Regression - Average Results ===")
    print(f"Average RMSE (using optimal k for each fold): {avg_rmse:.4f}")
    print(f"Average MAE (using optimal k for each fold): {avg_mae:.4f}")
    print(f"Average optimal k across folds: {avg_k:.1f}")
    print(f"Optimal k values by fold: {best_k_values}")
    
    # Create visualizations
    plt.figure(figsize=(14, 6))
    
    # Plot metrics
    bar_width = 0.35
    x = results_df['fold']
    x_pos = np.arange(len(x))
    
    # Create bars
    plt.bar(x_pos - bar_width/2, results_df['rmse'], bar_width, label='RMSE', color='blue', alpha=0.7)
    plt.bar(x_pos + bar_width/2, results_df['mae'], bar_width, label='MAE', color='red', alpha=0.7)
    
    # Add fold numbers to x-axis
    plt.xticks(x_pos, results_df['fold'])
    
    # Add value labels on top of bars
    for i, (rmse, mae, k) in enumerate(zip(results_df['rmse'], results_df['mae'], results_df['k'])):
        plt.text(i - bar_width/2, rmse + 0.005, f'{rmse:.4f} (k={k})', ha='center', va='bottom', fontsize=8)
        plt.text(i + bar_width/2, mae + 0.005, f'{mae:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Fold')
    plt.ylabel('Error Value (RMSE / MAE)')
    plt.title('KNN Regression Performance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save visualization
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig('visualizations/KNN_Regression_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot k values by fold
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(1, n_splits + 1), best_k_values, color='purple', alpha=0.7)
    plt.axhline(y=avg_k, color='red', linestyle='--', label=f'Average: {avg_k:.1f}')
    
    # Add k value labels on top of bars
    for i, (bar, k) in enumerate(zip(bars, best_k_values)):
        plt.text(i+1, k + 1, f'k={k}', ha='center', va='bottom')
    
    plt.xlabel('Fold')
    plt.ylabel('Optimal k')
    plt.title('Optimal k Values by Fold')
    plt.xticks(range(1, n_splits + 1))
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/KNN_Regression_k_values.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a visualization of k vs RMSE
    if len(k_values) > 5:  # Only if we have multiple k values to compare
        plt.figure(figsize=(12, 6))
        
        # For demonstration, run a detailed k analysis on the last fold
        last_fold = n_splits - 1
        train_idx = list(list(tscv.split(X))[last_fold])[0]
        test_idx = list(list(tscv.split(X))[last_fold])[1]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Try different k values
        k_values_detailed = list(range(1, 51))
        rmse_values = []
        
        for k in k_values_detailed:
            knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
            knn.fit(X_train_scaled, y_train)
            predictions = knn.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            rmse_values.append(rmse)
        
        # Plot RMSE vs k
        plt.plot(k_values_detailed, rmse_values, marker='o', linestyle='-', color='blue')
        plt.axvline(x=best_k_values[last_fold], color='red', linestyle='--', 
                    label=f'Selected k={best_k_values[last_fold]}')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('RMSE')
        plt.title('RMSE vs k for Last Fold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('visualizations/KNN_Regression_k_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"KNN regression visualizations saved to 'visualizations/' directory")
    
    return results_df, best_k_values

def ffnn_regression_cv(data, n_splits=5, verbose=False):
    """
    Perform Feed-Forward Neural Network regression with time series cross-validation.
    Uses scikit-learn's MLPRegressor for neural network implementation.
    
    Parameters:
    - data: DataFrame containing features and target variable
    - n_splits: Number of splits for time series cross-validation
    - verbose: If True, print results for each fold
    
    Returns:
    - Results DataFrame and optimized network parameters
    """
    print("\nPerforming Feed-Forward Neural Network with cross-validation...")
    
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
    
    # Define hyperparameter options
    hidden_layer_sizes_options = [(16,), (32,), (64,), (16, 8), (32, 16), (64, 32)]
    alpha_options = [0.0001, 0.001, 0.01, 0.1]
    learning_rate_options = ['constant', 'adaptive']
    
    # Use TimeSeriesSplit for time-ordered cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    best_params_all_folds = []
    
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
        
        if verbose:
            print(f"\nFold {fold+1}")
            print(f"Train: {train_start} to {train_end} ({len(X_train)} samples)")
            print(f"Test: {test_start} to {test_end} ({len(X_test)} samples)")
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Grid search for optimal hyperparameters
        param_grid = {
            'hidden_layer_sizes': hidden_layer_sizes_options,
            'alpha': alpha_options,
            'learning_rate': learning_rate_options
        }
        
        # Perform nested cross-validation to find best hyperparameters
        inner_cv = TimeSeriesSplit(n_splits=3)  # Smaller number of splits for inner CV
        mlp = MLPRegressor(
            max_iter=1000, 
            random_state=42, 
            activation='relu',
            early_stopping=True,
            validation_fraction=0.1
        )
        
        grid = GridSearchCV(
            estimator=mlp, 
            param_grid=param_grid, 
            cv=inner_cv, 
            verbose=0, 
            n_jobs=-1, 
            scoring='neg_mean_squared_error'
        )
        
        if verbose:
            print("Finding optimal hyperparameters...")
            
        grid_result = grid.fit(X_train_scaled, y_train)
        
        # Get best parameters
        best_params = grid_result.best_params_
        best_params_all_folds.append(best_params)
        if verbose:
            print(f"Best parameters for fold {fold+1}: {best_params}")
        
        # Train with best parameters
        best_mlp = MLPRegressor(
            hidden_layer_sizes=best_params['hidden_layer_sizes'],
            alpha=best_params['alpha'],
            learning_rate=best_params['learning_rate'],
            activation='relu',
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        best_mlp.fit(X_train_scaled, y_train)
        
        # Predict on test set
        predictions = best_mlp.predict(X_test_scaled)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        if verbose:
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        # Store results
        results.append({
            'fold': fold+1,
            'hidden_layers': str(best_params['hidden_layer_sizes']),
            'alpha': best_params['alpha'],
            'learning_rate': best_params['learning_rate'],
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
    
    print("\n=== Feed-Forward Neural Network - Average Results ===")
    print(f"Average RMSE: {avg_rmse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    
    # Print hyperparameters
    print("\n=== Neural Network Hyperparameters by Fold ===")
    for i, params in enumerate(best_params_all_folds):
        print(f"Fold {i+1}: Hidden Layers={params['hidden_layer_sizes']}, Alpha={params['alpha']}, Learning Rate={params['learning_rate']}")
    
    # Create visualizations
    create_nn_visualizations(results_df, best_params_all_folds)
    
    return results_df, best_params_all_folds

def create_nn_visualizations(results_df, best_params_all_folds):
    """
    Create visualizations for Neural Network results.
    
    Parameters:
    - results_df: DataFrame with fold results
    - best_params_all_folds: List of best hyperparameters for each fold
    """
    # 1. Performance by fold
    plt.figure(figsize=(14, 6))
    
    # Plot metrics
    bar_width = 0.35
    x = results_df['fold']
    x_pos = np.arange(len(x))
    
    # Create bars
    plt.bar(x_pos - bar_width/2, results_df['rmse'], bar_width, label='RMSE', color='blue', alpha=0.7)
    plt.bar(x_pos + bar_width/2, results_df['mae'], bar_width, label='MAE', color='red', alpha=0.7)
    
    # Add fold numbers to x-axis
    plt.xticks(x_pos, results_df['fold'])
    
    # Add value labels on top of bars
    for i, (rmse, mae, hidden, alpha, lr) in enumerate(zip(
            results_df['rmse'], results_df['mae'], 
            results_df['hidden_layers'], results_df['alpha'],
            results_df['learning_rate'])):
        plt.text(i - bar_width/2, rmse + 0.005, f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)
        plt.text(i + bar_width/2, mae + 0.005, f'{mae:.4f}', ha='center', va='bottom', fontsize=9)
        plt.text(i, 0.01, f'{hidden}\nα={alpha}\n{lr}', ha='center', va='bottom', fontsize=8)
    
    plt.xlabel('Fold')
    plt.ylabel('Error Value (RMSE / MAE)')
    plt.title('Neural Network Performance by Fold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    plt.savefig('visualizations/NN_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Neural Network visualizations saved to 'visualizations/' directory")

def compare_all_models(ols_results, ridge_results, lasso_results, knn_results, nn_results=None):
    """
    Compare the performance of OLS, Ridge, Lasso, KNN, and Neural Network regression models.
    
    Parameters:
    - ols_results: DataFrame with OLS results
    - ridge_results: DataFrame with Ridge results
    - lasso_results: DataFrame with Lasso results
    - knn_results: DataFrame with KNN results
    - nn_results: DataFrame with Neural Network results (optional)
    
    Returns:
    - Comparison DataFrame and average performance DataFrame
    """
    # Prepare data for comparison
    ols_rmse = ols_results['rmse']
    ridge_rmse = ridge_results['rmse']
    lasso_rmse = lasso_results['rmse']
    knn_rmse = knn_results['rmse']
    
    ols_mae = ols_results['mae']
    ridge_mae = ridge_results['mae']
    lasso_mae = lasso_results['mae']
    knn_mae = knn_results['mae']
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Fold': range(1, len(ols_rmse) + 1),
        'OLS_RMSE': ols_rmse,
        'Ridge_RMSE': ridge_rmse,
        'Lasso_RMSE': lasso_rmse,
        'KNN_RMSE': knn_rmse,
        'OLS_MAE': ols_mae,
        'Ridge_MAE': ridge_mae,
        'Lasso_MAE': lasso_mae,
        'KNN_MAE': knn_mae
    })
    
    # Add neural network results if provided
    model_names = ['OLS', 'Ridge', 'Lasso', 'KNN']
    if nn_results is not None and not nn_results.empty:
        nn_rmse = nn_results['rmse']
        nn_mae = nn_results['mae']
        comparison_df['NN_RMSE'] = nn_rmse
        comparison_df['NN_MAE'] = nn_mae
        model_names.append('NN')
    
    # Create visualization comparing all models
    plt.figure(figsize=(16, 8))
    
    # Plot RMSE comparison
    plt.subplot(1, 2, 1)
    x = comparison_df['Fold']
    x_pos = np.arange(len(x))
    
    num_models = len(model_names)
    bar_width = 0.8 / num_models
    
    colors = {
        'OLS': 'blue',
        'Ridge': 'green',
        'Lasso': 'orange',
        'KNN': 'purple',
        'NN': 'red'
    }
    
    # Position multipliers for bar placement
    positions = np.linspace(-(num_models-1)/2, (num_models-1)/2, num_models)
    
    # Create RMSE bars
    for i, model in enumerate(model_names):
        plt.bar(x_pos + positions[i]*bar_width, 
                comparison_df[f'{model}_RMSE'], 
                bar_width, 
                label=model, 
                color=colors[model], 
                alpha=0.7)
    
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title('RMSE Comparison by Fold')
    plt.xticks(x_pos, comparison_df['Fold'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot MAE comparison
    plt.subplot(1, 2, 2)
    
    # Create MAE bars
    for i, model in enumerate(model_names):
        plt.bar(x_pos + positions[i]*bar_width, 
                comparison_df[f'{model}_MAE'], 
                bar_width, 
                label=model, 
                color=colors[model], 
                alpha=0.7)
    
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    plt.title('MAE Comparison by Fold')
    plt.xticks(x_pos, comparison_df['Fold'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/all_models_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate average performance
    avg_performance = {
        'Model': model_names,
        'Avg RMSE': [
            ols_rmse.mean(), 
            ridge_rmse.mean(), 
            lasso_rmse.mean(), 
            knn_rmse.mean()
        ],
        'Avg MAE': [
            ols_mae.mean(), 
            ridge_mae.mean(), 
            lasso_mae.mean(), 
            knn_mae.mean()
        ]
    }
    
    # Add NN to average performance if provided
    if nn_results is not None and not nn_results.empty:
        avg_performance['Avg RMSE'].append(nn_rmse.mean())
        avg_performance['Avg MAE'].append(nn_mae.mean())
    
    avg_df = pd.DataFrame(avg_performance)
    
    # Create bar chart of average performance
    plt.figure(figsize=(12, 6))
    x_pos = np.arange(len(avg_df['Model']))
    bar_width = 0.35
    
    plt.bar(x_pos - bar_width/2, avg_df['Avg RMSE'], bar_width, label='RMSE', color='blue', alpha=0.7)
    plt.bar(x_pos + bar_width/2, avg_df['Avg MAE'], bar_width, label='MAE', color='red', alpha=0.7)
    
    # Add value labels on top of bars
    for i, (rmse, mae) in enumerate(zip(avg_df['Avg RMSE'], avg_df['Avg MAE'])):
        plt.text(i - bar_width/2, rmse + 0.001, f'{rmse:.4f}', ha='center', va='bottom')
        plt.text(i + bar_width/2, mae + 0.001, f'{mae:.4f}', ha='center', va='bottom')
    
    plt.xlabel('Model')
    plt.ylabel('Error Value')
    plt.title('Average Model Performance')
    plt.xticks(x_pos, avg_df['Model'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualizations/average_all_models_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n=== Model Comparison ===")
    print(avg_df.to_string(index=False, float_format='%.4f'))
    
    # Determine the best model based on RMSE
    best_model_idx = np.argmin(avg_df['Avg RMSE'])
    best_model = avg_df.loc[best_model_idx, 'Model']
    
    print(f"\nBest performing model based on RMSE: {best_model}")
    
    return comparison_df, avg_df

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
    
    # Drop rows with NaN values before evaluation
    combined_data_clean = combined_data.dropna()
    
    # Evaluate OLS regression
    ols_results, ols_coefs = evaluate_model_metrics(combined_data_clean, verbose=False)
    
    # Evaluate ridge regression with automatic alpha selection
    ridge_results, ridge_coefs, ridge_alphas = ridge_regression_cv(combined_data_clean, verbose=False)
    
    # Evaluate lasso regression with automatic alpha selection
    lasso_results, lasso_coefs, lasso_alphas = lasso_regression_cv(combined_data_clean, verbose=False)
    
    # Evaluate KNN regression with automatic k selection
    knn_results, knn_k_values = knn_regression_cv(combined_data_clean, verbose=False)
    
    # Evaluate Feed-Forward Neural Network
    nn_results, nn_params = ffnn_regression_cv(combined_data_clean, verbose=False)
    
    # Compare all models
    comparison_df, avg_performance = compare_all_models(ols_results, ridge_results, lasso_results, knn_results, nn_results)
    
    # Save results to pickle files for later use in comparison plots
    if not os.path.exists('results_cache'):
        os.makedirs('results_cache')
    
    # Add k values to knn_results for better comparison plots
    knn_results['k'] = knn_k_values
    
    with open('results_cache/ols_results.pkl', 'wb') as f:
        pickle.dump(ols_results, f)
    with open('results_cache/ridge_results.pkl', 'wb') as f:
        pickle.dump(ridge_results, f)
    with open('results_cache/lasso_results.pkl', 'wb') as f:
        pickle.dump(lasso_results, f)
    with open('results_cache/knn_results.pkl', 'wb') as f:
        pickle.dump(knn_results, f)
    with open('results_cache/nn_results.pkl', 'wb') as f:
        pickle.dump(nn_results, f)
