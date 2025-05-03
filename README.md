# Equity Volatility Forecasting

This project compares different machine learning models for predicting future equity volatility using features from S&P 500 and VIX data.

## Requirements

- Python 3.8+
- Required libraries listed in `requirements.txt`

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd equity_vol
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create required directories:
```bash
mkdir -p data_cache visualizations results_cache
```

## Running the Code

Run the main script to download data, train models, and generate visualizations:

```bash
python main.py
```


## Models Implemented

The code implements and compares several regression models:

1. **VIX Baseline** - Using VIX directly as prediction
2. **OLS (Ordinary Least Squares)** - Linear regression with multiple features
3. **Ridge Regression** - Linear model with L2 regularization
4. **Lasso Regression** - Linear model with L1 regularization
5. **KNN (K-Nearest Neighbors)** - Non-parametric regression
6. **Neural Network** - Feed-forward neural network with optimized hyperparameters

## Features

- Historical returns (1-week, 3-month)
- Historical volatility measures (1-week, 3-month)
- Volume changes (1-week, 3-month)
- VIX index (implied volatility)

## Output

The script will generate:

1. Visual comparison of all models' performance
2. Feature importance analysis
3. Detailed metrics (RMSE, MAE) for each model
4. Hyperparameter optimization results

All visualizations are saved in the `visualizations/` directory.
