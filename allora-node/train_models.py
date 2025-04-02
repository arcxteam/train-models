import numpy as np
import sqlite3
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import logging
import zipfile
import glob
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import json
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Import app_config for paths
from app_config import DATABASE_PATH, DATA_BASE_PATH

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
BINANCE_DIR = os.path.join(DATA_BASE_PATH, 'binance', 'futures-klines')

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Function to download token data from Binance
def download_token_data(token_name, interval="1m", months=1):
    """
    Download token data from Binance for the specified months
    
    Args:
        token_name (str): Token name in lowercase (e.g., 'bera', 'eth')
        interval (str): Data interval (default: '1m' for 1 minute)
        months (int): Number of months of data to download
    """
    logger.info(f"Downloading {token_name.upper()} data for the last {months} months")
    
    # Convert token to proper Binance format
    binance_token = f"{token_name.upper()}USDT"
    
    # Create directory if not exists
    token_dir = os.path.join(BINANCE_DIR, binance_token.lower())
    os.makedirs(token_dir, exist_ok=True)
    
    # Current date
    current_date = datetime.now()
    
    # Download data for the specified months
    for i in range(months):
        target_date = current_date - timedelta(days=30*i)
        year = target_date.year
        month = target_date.month
        
        # Base URL for Binance data
        base_url = f"https://data.binance.vision/data/futures/um/daily/klines/{binance_token}/{interval}"
        
        # Download for each day of the month
        for day in range(1, 32):
            # Skip invalid days
            if day > 28:
                if month == 2 and (year % 4 != 0 or (year % 100 == 0 and year % 400 != 0)):
                    continue  # Skip Feb 29-31 in non-leap years
                elif month == 2 and day > 29:
                    continue  # Skip Feb 30-31 in leap years
                elif month in [4, 6, 9, 11] and day > 30:
                    continue  # Skip day 31 in months with 30 days
            
            # URL for specific day
            url = f"{base_url}/{binance_token}-{interval}-{year}-{month:02d}-{day:02d}.zip"
            
            # Target file path
            zip_file = os.path.join(token_dir, f"{binance_token}-{interval}-{year}-{month:02d}-{day:02d}.zip")
            
            # Skip if already downloaded
            if os.path.exists(zip_file):
                continue
            
            try:
                # Download file
                response = requests.get(url)
                if response.status_code == 200:
                    with open(zip_file, 'wb') as f:
                        f.write(response.content)
                    logger.info(f"Downloaded {url}")
                else:
                    logger.warning(f"Failed to download {url}: {response.status_code}")
            except Exception as e:
                logger.error(f"Error downloading {url}: {e}")

# Function to extract ZIP files
def extract_zip_files(token_name):
    # Convert token to proper Binance format
    binance_token = token_name.replace('usd', 'usdt').lower()
    token_dir = os.path.join(BINANCE_DIR, binance_token)
    
    if not os.path.exists(token_dir):
        logger.warning(f"Directory not found: {token_dir}")
        return False
    
    # Get all ZIP files
    zip_files = glob.glob(os.path.join(token_dir, "*.zip"))
    if not zip_files:
        logger.warning(f"No ZIP files found in {token_dir}")
        return False
    
    logger.info(f"Found {len(zip_files)} ZIP files for {token_name}")
    
    # Extract each ZIP file
    for zip_file in zip_files:
        csv_file = zip_file.replace('.zip', '.csv')
        # Skip if CSV already exists
        if os.path.exists(csv_file):
            continue
        
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Get the first file in the ZIP (should be the CSV)
                csv_name = zip_ref.namelist()[0]
                # Extract to the same directory with renamed filename
                zip_ref.extract(csv_name, token_dir)
                # Rename to match ZIP filename but with .csv extension
                extracted_path = os.path.join(token_dir, csv_name)
                os.rename(extracted_path, csv_file)
                logger.info(f"Extracted: {zip_file} to {csv_file}")
        except Exception as e:
            logger.error(f"Error extracting {zip_file}: {e}")
    
    return True

# Combine all extracted CSVs for a token
def combine_csv_files(token_name):
    # Convert token to proper Binance format
    binance_token = token_name.replace('usd', 'usdt').lower()
    token_dir = os.path.join(BINANCE_DIR, binance_token)
    
    if not os.path.exists(token_dir):
        logger.warning(f"Directory not found: {token_dir}")
        return None
    
    # Get all CSV files
    csv_files = glob.glob(os.path.join(token_dir, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {token_dir}")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files for {token_name}")
    
    # Read and combine all CSVs
    all_data = []
    for csv_file in sorted(csv_files):
        try:
            # Read CSV (Binance format)
            df = pd.read_csv(csv_file, header=None)
            df.columns = ["open_time", "open", "high", "low", "close", 
                          "volume", "close_time", "quote_volume", 
                          "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
            all_data.append(df)
        except Exception as e:
            logger.error(f"Error reading {csv_file}: {e}")
    
    if not all_data:
        logger.warning(f"No data loaded from CSVs for {token_name}")
        return None
    
    # Combine all dataframes
    combined_df = pd.concat(all_data)
    
    # Sort by open_time
    combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
    
    # Convert timestamps from milliseconds to seconds
    combined_df['timestamp'] = combined_df['open_time'] / 1000
    
    # Create a simplified dataframe with just price and timestamp
    simplified_df = pd.DataFrame({
        'price': combined_df['close'],
        'timestamp': combined_df['timestamp'],
        'open': combined_df['open'],
        'high': combined_df['high'],
        'low': combined_df['low'],
        'volume': combined_df['volume']
    })
    
    return simplified_df

# Fetch data from the database
def load_data_from_db(token_name, hours_back=None):
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if timestamp column exists
            cursor.execute(f"PRAGMA table_info(prices)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'timestamp' not in columns:
                # Query without timestamp column
                if hours_back:
                    # Get the latest N prices
                    cursor.execute("""
                        SELECT price FROM prices 
                        WHERE token=?
                        ORDER BY block_height DESC
                        LIMIT ?
                    """, (token_name.lower(), hours_back * 60))  # Assuming 1 price per minute
                else:
                    cursor.execute("""
                        SELECT price FROM prices 
                        WHERE token=?
                        ORDER BY block_height DESC
                    """, (token_name.lower(),))
                    
                result = cursor.fetchall()
                
                if result:
                    prices = np.array([x[0] for x in result])
                    # Create dummy timestamps (we don't have real ones)
                    timestamps = np.arange(len(prices))
                    # Reverse to make chronological
                    return prices[::-1].reshape(-1, 1), timestamps[::-1]
            else:
                # Original query with timestamp
                if hours_back:
                    cursor.execute("""
                        SELECT price, timestamp FROM prices 
                        WHERE token=? AND timestamp >= (SELECT MAX(timestamp) FROM prices WHERE token=?) - ? * 3600
                        ORDER BY block_height ASC
                    """, (token_name.lower(), token_name.lower(), hours_back))
                else:
                    cursor.execute("""
                        SELECT price, timestamp FROM prices 
                        WHERE token=?
                        ORDER BY block_height ASC
                    """, (token_name.lower(),))
                    
                result = cursor.fetchall()
                
                if result:
                    prices = np.array([x[0] for x in result]).reshape(-1, 1)
                    timestamps = np.array([x[1] for x in result])
                    return prices, timestamps
        
        logger.warning(f"No data found in database for {token_name}")
        return None, None
    
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        return None, None

# Load data from directly from Binance CSV files
def load_data_from_binance(token_name, hours_back=None):
    logger.info(f"Loading data from Binance files for {token_name}")
    
    # First extract all ZIP files
    extraction_successful = extract_zip_files(token_name)
    if not extraction_successful:
        logger.warning(f"Could not extract ZIP files for {token_name}")
    
    # Combine all CSV files
    df = combine_csv_files(token_name)
    if df is None:
        logger.warning(f"Could not combine CSV files for {token_name}")
        return None, None
    
    # Filter by hours_back if specified
    if hours_back:
        # Calculate cutoff timestamp
        if len(df) > 0:
            max_timestamp = df['timestamp'].max()
            min_timestamp = max_timestamp - (hours_back * 3600)
            df = df[df['timestamp'] >= min_timestamp]
    
    # Get numpy arrays
    if len(df) > 0:
        prices = df['price'].values.reshape(-1, 1)
        timestamps = df['timestamp'].values
        logger.info(f"Loaded {len(prices)} data points from Binance files for {token_name}")
        return prices, timestamps
    
    logger.warning(f"No data loaded from Binance files for {token_name}")
    return None, None

# Combined data loading function that tries multiple sources
def load_data(token_name, hours_back=None):
    """
    Load data for the specified token, prioritizing CSV data over database
    """
    # First try loading from Binance files (prioritize this)
    logger.info(f"Trying to load data from Binance CSV files for {token_name}")
    prices, timestamps = load_data_from_binance(token_name, hours_back)
    
    # If CSV data not found, try database as fallback
    if prices is None:
        logger.info(f"Data not found in CSV for {token_name}, trying database...")
        prices, timestamps = load_data_from_db(token_name, hours_back)
    
    if prices is None:
        raise ValueError(f"Could not load data for {token_name} from any source")
    
    logger.info(f"Loaded {len(prices)} data points for {token_name}")
    return prices, timestamps

#-----------------------------------------------------------------------
# Functions for BERA Log-Return Prediction
#-----------------------------------------------------------------------

def calculate_log_returns(prices, period=60):
    """
    Calculate log returns for the given prices with specified period.
    
    Args:
        prices (numpy.array): Array of price data
        period (int): Period for calculating returns in data points (default: 60 for 1 hour with 1-minute data)
        
    Returns:
        numpy.array: Array of log returns
    """
    # Ensure prices is a flattened array
    if prices.ndim > 1:
        prices = prices.flatten()
    
    # Calculate log returns: ln(price_t+period / price_t)
    log_returns = np.log(prices[period:] / prices[:-period])
    
    return log_returns

def calculate_mztae(y_true, y_pred, sigma=None):
    """
    Calculate Modified Z-transformed Absolute Error.
    
    Args:
        y_true (numpy.array): Actual values
        y_pred (numpy.array): Predicted values
        sigma (float, optional): Standard deviation for normalization, if None calculated from data
        
    Returns:
        float: MZTAE score
    """
    # Ensure arrays are 1D
    y_true = y_true.flatten() if hasattr(y_true, 'flatten') else y_true
    y_pred = y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred
    
    # Calculate absolute errors
    abs_errors = np.abs(y_true - y_pred)
    
    # Initialize array for Z-transformed errors
    ztae_values = np.zeros_like(abs_errors)
    
    # Use rolling window of 100 for standard deviation calculation
    window_size = 100
    
    for i in range(len(abs_errors)):
        # Define the window start (ensure we don't go below 0)
        window_start = max(0, i - window_size)
        
        # Get the ground truth values for this window
        window_values = y_true[window_start:i]
        
        # If we have enough values in the window, use them for sigma
        # Otherwise, use all available values or provided sigma
        if len(window_values) > 0:
            sigma_i = np.std(window_values)
        elif sigma is not None:
            sigma_i = sigma
        else:
            sigma_i = np.std(y_true[:i+1]) if i > 0 else 1e-8
        
        # Avoid division by zero
        if sigma_i < 1e-8:
            sigma_i = 1e-8
        
        # Calculate Z-transformed error
        ztae_values[i] = np.tanh(abs_errors[i] / sigma_i)
    
    # Calculate mean ZTAE
    mztae = np.mean(ztae_values)
    
    return mztae

def smooth_predictions(preds, alpha=0.2):
    """ Exponential Moving Average (EMA) smoothing for post-prediction """
    smoothed = [preds[0]]
    for p in preds[1:]:
        smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def prepare_data_for_log_return(data, look_back, prediction_period):
    """
    Prepare data for training log-return prediction models.
    
    Args:
        data (numpy.array): Price data
        look_back (int): Number of data points to use for prediction
        prediction_period (int): Prediction horizon in data points
        
    Returns:
        X, Y, prev_Y, scaler: Prepared training data and scaler
    """
    # Ensure data is a flattened array
    if data.ndim > 1:
        data = data.flatten()
    
    # Calculate log returns for the entire series
    # We'll use these to create our target values
    all_log_returns = calculate_log_returns(data, prediction_period)
    
    # We need to adjust our data range to align with log returns
    # We lose `prediction_period` data points when calculating log returns
    adjusted_data = data[:-prediction_period]
    
    # Scale the price data
    price_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = price_scaler.fit_transform(adjusted_data.reshape(-1, 1))
    
    # Prepare feature (X) and target (Y) data
    X, Y, prev_Y = [], [], []
    
    # We need at least look_back + 1 points to create a sample
    for i in range(len(scaled_prices) - look_back):
        # X is a window of scaled prices
        X.append(scaled_prices[i:i + look_back, 0])
        # Y is the log return for the prediction period starting at the end of X
        Y.append(all_log_returns[i])
        # prev_Y is the last price in the window (for directional accuracy)
        prev_Y.append(scaled_prices[i + look_back - 1, 0])
    
    X = np.array(X)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    
    return X, Y, prev_Y, price_scaler

def prepare_data_for_log_return_lstm(data, look_back, prediction_period):
    """
    Prepare data for training LSTM log-return prediction models.
    """
    X, Y, prev_Y, scaler = prepare_data_for_log_return(data, look_back, prediction_period)
    
    # Reshape X for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y, prev_Y, scaler
    
def prepare_data_for_log_return_enhanced(data, look_back, prediction_period):
    """
    Prepare data for training log-return prediction models with OHLC data.
    
    Args:
        data (pandas.DataFrame): DataFrame with price, open, high, low, volume data
        look_back (int): Number of data points to use for prediction
        prediction_period (int): Prediction horizon in data points
        
    Returns:
        X, Y, prev_Y, scaler: Prepared training data and scaler
    """
    # If data is a numpy array, convert to DataFrame
    if isinstance(data, np.ndarray):
        # Ensure the data is flattened
        if data.ndim > 1:
            data = data.flatten()
        # Create a DataFrame with just the price column
        df = pd.DataFrame({'price': data})
    else:
        # Use the DataFrame as is
        df = data.copy()
    
    # Ensure all necessary columns exist
    required_cols = ['price']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column {col} not found in data")
    
    # Calculate log returns for the target variable
    # We use the price column for this
    price_array = df['price'].values
    log_returns = calculate_log_returns(price_array, prediction_period)
    
    # Calculate additional features based on OHLC data if available
    features_list = ['price']
    
    # Add OHLC columns if available
    if 'open' in df.columns and 'high' in df.columns and 'low' in df.columns:
        features_list.extend(['open', 'high', 'low'])
        
        # Add some derived features
        # Candle body
        df['body_size'] = abs(df['close'] - df['open']) / df['open']
        features_list.append('body_size')
        
        # Upper and lower shadows
        df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
        df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
        features_list.extend(['upper_shadow', 'lower_shadow'])
    
    # Add volume if available
    if 'volume' in df.columns:
        features_list.append('volume')
        
        # Add volume-based features
        df['volume_change'] = df['volume'].pct_change()
        features_list.append('volume_change')
    
    # Add simple moving averages
    if len(df) > 20:
        df['ema_5'] = df['price'].rolling(window=5).mean()
        df['ema_10'] = df['price'].rolling(window=10).mean()
        df['ema_20'] = df['price'].rolling(window=20).mean()
        features_list.extend(['ema_5', 'ema_10', 'ema_20'])
    
    # Fill NaN values
    df = df.fillna(method='bfill').fillna(0)
    
    # We need to adjust our data range to align with log returns
    # We lose `prediction_period` data points when calculating log returns
    df_adjusted = df.iloc[:-prediction_period].copy()
    
    # Scale the features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(df_adjusted[features_list])
    
    # Prepare feature (X) and target (Y) data
    X, Y, prev_Y = [], [], []
    
    # We need at least look_back + 1 points to create a sample
    for i in range(len(scaled_features) - look_back):
        # X is a window of scaled features
        X.append(scaled_features[i:i + look_back])
        # Y is the log return for the prediction period
        Y.append(log_returns[i])
        # prev_Y is the last price in the window (for directional accuracy)
        prev_Y.append(scaled_features[i + look_back - 1, 0])  # price is at index 0
    
    X = np.array(X)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    
    return X, Y, prev_Y, scaler

def train_and_save_log_return_model(token_name, look_back, prediction_horizon, hours_data=None, model_type='rf'):
    """
    Train and save a model for log-return prediction
    
    Args:
        token_name (str): Token name (e.g., 'berausd')
        look_back (int): Number of past data points to use for prediction
        prediction_horizon (int): Prediction horizon in minutes
        hours_data (int): Hours of historical data to use
        model_type (str): Model type to use ('rf', 'xgb', or 'lstm')
    """
    try:
        logger.info(f"Training {model_type.upper()} Log-Return model for {token_name} with {prediction_horizon}-minute horizon")
        
        # Load data
        data, timestamps = load_data(token_name, hours_data)
        
        # Convert data to DataFrame if it's a numpy array
        if isinstance(data, np.ndarray):
            # We need to reshape to ensure it's a proper column
            if data.ndim > 1:
                price_data = data.flatten()
            else:
                price_data = data
            data_df = pd.DataFrame({'price': price_data, 'timestamp': timestamps})
        else:
            data_df = data

        # Prepare data based on model type
        if model_type == 'lstm':
            # For LSTM, we need to reshape the data differently
            X, Y, prev_Y, scaler = prepare_data_for_log_return_lstm(data_df['price'].values, look_back, prediction_horizon)
        else:
            # Use the enhanced preparation function that can handle OHLC data
            X, Y, prev_Y, scaler = prepare_data_for_log_return_enhanced(data_df, look_back, prediction_horizon)
        
        # Use a portion for final testing
        X_train_val, X_test, Y_train_val, Y_test, prev_Y_train_val, prev_Y_test = train_test_split(
            X, Y, prev_Y, test_size=0.2, shuffle=False
        )
        
        # Use time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        if model_type == 'rf':
            # Random Forest model
            pipeline = Pipeline([('rf', RandomForestRegressor(random_state=42))])
            
            # Hyperparameter grid - reduced for faster training
            param_dist = {
                'rf__n_estimators': [100, 200, 300],
                'rf__max_depth': [None, 10, 20],
                'rf__min_samples_split': [2, 5],
                'rf__min_samples_leaf': [1, 2],
                'rf__max_features': ['sqrt', 'log2']
            }
            
            # Use RandomizedSearchCV with TimeSeriesSplit
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_dist, n_iter=5, cv=tscv,
                scoring='neg_mean_absolute_error', n_jobs=3, random_state=42, verbose=1
            )
            
            # Fit the model
            search.fit(X_train_val, Y_train_val)
            best_model = search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate regular metrics
            mae = mean_absolute_error(Y_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test, y_pred)
            
            # Calculate MZTAE using rolling window approach
            mztae_score = calculate_mztae(Y_test, y_pred)
            
            # Calculate directional accuracy where possible
            # For log returns, we compare the sign of the predicted vs actual log return
            dir_acc = 100 * np.mean((Y_test > 0) == (y_pred > 0))
            
            # Log results
            logger.info(f"RF Log-Return Model - MAE: {mae}")
            logger.info(f"RF Log-Return Model - RMSE: {rmse}")
            logger.info(f"RF Log-Return Model - R² Score: {r2}")
            logger.info(f"RF Log-Return Model - MZTAE Score: {mztae_score}")
            logger.info(f"RF Log-Return Model - Directional Accuracy: {dir_acc:.2f}%")
            
            # Save model and scaler
            model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_rf_model_{prediction_horizon}m.pkl')
            joblib.dump(best_model, model_path)
            
            scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_scaler_{prediction_horizon}m.pkl')
            joblib.dump(scaler, scaler_path)
            
            # Save metrics
            metrics = {
                'model': f'{model_type}_logreturn',
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mztae': float(mztae_score),
                'directional_accuracy': float(dir_acc)
            }
            
            metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_rf_metrics_{prediction_horizon}m.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            logger.info(f"RF Log-Return Model for {token_name} saved to {model_path}")
            
            return metrics
            
        elif model_type == 'xgb':
            # XGBoost model
            pipeline = Pipeline([('xgb', XGBRegressor(random_state=42, objective='reg:squarederror'))])
            
            # Hyperparameter grid for XGBoost - reduced for faster training
            param_dist = {
                'xgb__n_estimators': [100, 200, 300],
                'xgb__max_depth': [3, 5, 7],
                'xgb__learning_rate': [0.01, 0.05, 0.1],
                'xgb__subsample': [0.7, 0.8, 0.9],
                'xgb__colsample_bytree': [0.7, 0.8, 0.9],
                'xgb__min_child_weight': [1, 2, 3],
                'xgb__gamma': [0, 0.1]
            }
            
            # Use RandomizedSearchCV with TimeSeriesSplit
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_dist, n_iter=5, cv=tscv,
                scoring='neg_mean_absolute_error', n_jobs=3, random_state=42, verbose=1
            )
            
            # Fit the model
            search.fit(X_train_val, Y_train_val)
            best_model = search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate regular metrics
            mae = mean_absolute_error(Y_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test, y_pred)
            
            # Calculate MZTAE using rolling window approach
            mztae_score = calculate_mztae(Y_test, y_pred)
            
            # Calculate directional accuracy
            dir_acc = 100 * np.mean((Y_test > 0) == (y_pred > 0))
            
            # Log results
            logger.info(f"XGB Log-Return Model - MAE: {mae}")
            logger.info(f"XGB Log-Return Model - RMSE: {rmse}")
            logger.info(f"XGB Log-Return Model - R² Score: {r2}")
            logger.info(f"XGB Log-Return Model - MZTAE Score: {mztae_score}")
            logger.info(f"XGB Log-Return Model - Directional Accuracy: {dir_acc:.2f}%")
            
            # Save model and scaler
            model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_xgb_model_{prediction_horizon}m.pkl')
            joblib.dump(best_model, model_path)
            
            scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_scaler_{prediction_horizon}m.pkl')
            joblib.dump(scaler, scaler_path)
            
            # Save metrics
            metrics = {
                'model': f'{model_type}_logreturn',
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mztae': float(mztae_score),
                'directional_accuracy': float(dir_acc)
            }
            
            metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_xgb_metrics_{prediction_horizon}m.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            logger.info(f"XGB Log-Return Model for {token_name} saved to {model_path}")
            
            return metrics
            
        elif model_type == 'lstm':
            # Split validation data from training data
            X_train, X_val = X_train_val[:int(0.8*len(X_train_val))], X_train_val[int(0.8*len(X_train_val)):]
            Y_train, Y_val = Y_train_val[:int(0.8*len(Y_train_val))], Y_train_val[int(0.8*len(Y_train_val)):]
            
            # Create LSTM model
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=30, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
            # Compile the model with Adam optimizer
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            # Callbacks for training
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            
            # Train the model
            history = model.fit(
                X_train, Y_train,
                epochs=50,  # Reduced for faster training
                batch_size=32,
                validation_data=(X_val, Y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            # Make predictions smoothing
            y_pred = model.predict(X_test).flatten()
            y_pred = smooth_predictions(y_pred, alpha=0.2)
            
            # Calculate regular metrics
            mae = mean_absolute_error(Y_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test, y_pred)
            
            # Calculate MZTAE using rolling window approach
            mztae_score = calculate_mztae(Y_test, y_pred)
            
            # Calculate directional accuracy
            dir_acc = 100 * np.mean((Y_test > 0) == (y_pred.flatten() > 0))
            
            # Log results
            logger.info(f"LSTM Log-Return Model - MAE: {mae}")
            logger.info(f"LSTM Log-Return Model - RMSE: {rmse}")
            logger.info(f"LSTM Log-Return Model - R² Score: {r2}")
            logger.info(f"LSTM Log-Return Model - MZTAE Score: {mztae_score}")
            logger.info(f"LSTM Log-Return Model - Directional Accuracy: {dir_acc:.2f}%")
            
            # Plot training & validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'LSTM Log-Return Model Loss for {token_name} ({prediction_horizon}m)')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lstm_loss_{prediction_horizon}m.png'))
            
            # Save model, scaler, and metrics
            model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_model_{prediction_horizon}m.keras')
            save_model(model, model_path)
            
            scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_scaler_{prediction_horizon}m.pkl')
            joblib.dump(scaler, scaler_path)
            
            # Save metrics
            metrics = {
                'model': f'{model_type}_logreturn',
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mztae': float(mztae_score),
                'directional_accuracy': float(dir_acc)
            }
            
            metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lstm_metrics_{prediction_horizon}m.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            logger.info(f"LSTM Log-Return Model for {token_name} saved to {model_path}")
            
            return metrics
            
    except Exception as e:
        logger.error(f"Error training log-return model for {token_name}: {e}", exc_info=True)
        return None

#-----------------------------------------------------------------------
# Functions for ETH Volatility Prediction
#-----------------------------------------------------------------------

def calculate_volatility(prices, window=360):
    """
    Calculate volatility as the standardized rolling standard deviation of log returns.
    
    Args:
        prices (numpy.array): Price data
        window (int): Window size for volatility calculation (default: 360 for 6 hours with 1-minute data)
        
    Returns:
        numpy.array: Volatility values
    """
    # Ensure prices is a flattened array
    if prices.ndim > 1:
        prices = prices.flatten()
    
    # Calculate log returns
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # Calculate rolling standard deviation
    volatility = []
    for i in range(len(log_returns) - window + 1):
        window_returns = log_returns[i:i+window]
        vol = np.std(window_returns)
        volatility.append(vol)
    
    # Convert to numpy array
    volatility = np.array(volatility)
    
    # Standardize (multiply by sqrt(360 / 1))
    # Since we're using 1-minute data, we multiply by sqrt(360)
    standardized_volatility = volatility * np.sqrt(360)
    
    return standardized_volatility

def prepare_data_for_volatility(data, look_back, prediction_horizon, window=360):
    """
    Prepare data for training volatility prediction models.
    
    Args:
        data (numpy.array): Price data
        look_back (int): Number of data points to use for prediction
        prediction_horizon (int): Prediction horizon in data points
        window (int): Window size for volatility calculation
        
    Returns:
        X, Y, prev_Y, scaler: Prepared training data and scaler
    """
    # Ensure data is a flattened array
    if data.ndim > 1:
        data = data.flatten()
    
    # Calculate volatility for the entire price series
    volatility = calculate_volatility(data, window)
    
    # We need to ensure our X, Y data align properly
    # The last point in our X data should be at index: len(data) - prediction_horizon - window
    # This ensures that the corresponding Y value (after prediction_horizon) has enough data
    # to calculate volatility using the window
    
    # Calculate how many points we lose due to volatility calculation and prediction horizon
    total_offset = window + prediction_horizon
    
    # Prepare the data, leaving enough points at the end for the prediction target
    X, Y, prev_Y = [], [], []
    
    for i in range(len(data) - look_back - total_offset + 1):
        # Features from the current window
        X.append(data[i:i+look_back])
        
        # Target is the volatility at prediction_horizon steps ahead
        Y.append(volatility[i+look_back+prediction_horizon-1])
        
        # Store the last price in the window for reference
        prev_Y.append(data[i+look_back-1])
    
    X = np.array(X)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    
    # Scale the input features
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_reshaped = X.reshape(X.shape[0], X.shape[1])
    X_scaled = scaler.fit_transform(X_reshaped)
    
    return X_scaled, Y, prev_Y, scaler

def prepare_data_for_volatility_lstm(data, look_back, prediction_horizon, window=360):
    """
    Prepare data for LSTM volatility prediction models.
    """
    X, Y, prev_Y, scaler = prepare_data_for_volatility(data, look_back, prediction_horizon, window)
    
    # Reshape X for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, Y, prev_Y, scaler

def train_and_save_volatility_model(token_name, look_back, prediction_horizon, hours_data=None, model_type='rf'):
    """
    Train and save model for volatility prediction
    
    Args:
        token_name (str): Token name (e.g., 'ethusd')
        look_back (int): Number of past data points to use for prediction
        prediction_horizon (int): Prediction horizon in minutes
        hours_data (int): Hours of historical data to use
        model_type (str): Model type to use ('rf', 'xgb', or 'lstm')
    """
    try:
        logger.info(f"Training {model_type.upper()} Volatility model for {token_name} with {prediction_horizon}-minute horizon")
        
        # Load data
        data, timestamps = load_data(token_name, hours_data)
        
        # Calculate the appropriate volatility window (360 minutes for 6h volatility)
        volatility_window = 360
        
        # Prepare data based on model type
        if model_type == 'lstm':
            X, Y, prev_Y, scaler = prepare_data_for_volatility_lstm(data, look_back, prediction_horizon, volatility_window)
        else:
            X, Y, prev_Y, scaler = prepare_data_for_volatility(data, look_back, prediction_horizon, volatility_window)
        
        # Use time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Use a portion for final testing (20% of data)
        total_samples = len(X)
        test_size = int(total_samples * 0.2)
        train_val_size = total_samples - test_size
        
        X_train_val = X[:train_val_size]
        Y_train_val = Y[:train_val_size]
        X_test = X[train_val_size:]
        Y_test = Y[train_val_size:]
        prev_Y_test = prev_Y[train_val_size:]
        
        if model_type == 'rf':
            # Random Forest for volatility prediction
            pipeline = Pipeline([('rf', RandomForestRegressor(random_state=42))])
            
            # Reduced parameter grid for faster training
            param_dist = {
                'rf__n_estimators': [100, 200, 300],
                'rf__max_depth': [None, 10, 20],
                'rf__min_samples_split': [2, 5],
                'rf__min_samples_leaf': [1, 2, 4],
                'rf__max_features': ['sqrt', 'log2', None]
            }
            
            # Use RandomizedSearchCV with TimeSeriesSplit
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_dist, n_iter=5, cv=tscv,
                scoring='neg_mean_squared_error', n_jobs=3, random_state=42, verbose=1
            )
            
            # Fit the model
            search.fit(X_train_val, Y_train_val)
            best_model = search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(Y_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test, y_pred)
            
            # Calculate MAPE
            mape = np.mean(np.abs((Y_test - y_pred) / np.maximum(Y_test, 1e-10))) * 100
            
            # Log results
            logger.info(f"RF Volatility Model - MAE: {mae}")
            logger.info(f"RF Volatility Model - RMSE: {rmse}")
            logger.info(f"RF Volatility Model - R² Score: {r2}")
            logger.info(f"RF Volatility Model - MAPE: {mape:.2f}%")
            
            # Save model and scaler
            model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_rf_model_{prediction_horizon}m.pkl')
            joblib.dump(best_model, model_path)
            
            scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_scaler_{prediction_horizon}m.pkl')
            joblib.dump(scaler, scaler_path)
            
            # Save metrics
            metrics = {
                'model': f'{model_type}_volatility',
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
            
            metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_rf_metrics_{prediction_horizon}m.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            logger.info(f"RF Volatility Model for {token_name} saved to {model_path}")
            
            return metrics
            
        elif model_type == 'xgb':
            # XGBoost for volatility prediction
            pipeline = Pipeline([('xgb', XGBRegressor(random_state=42, objective='reg:squarederror'))])
            
            # Parameter grid for XGBoost
            param_dist = {
                'xgb__n_estimators': [100, 200, 300],
                'xgb__max_depth': [3, 5, 7],
                'xgb__learning_rate': [0.01, 0.05, 0.1],
                'xgb__subsample': [0.7, 0.8, 0.9],
                'xgb__colsample_bytree': [0.7, 0.8, 0.9],
                'xgb__min_child_weight': [1, 2, 3],
                'xgb__gamma': [0, 0.1]
            }
            
            # RandomizedSearchCV with TimeSeriesSplit
            search = RandomizedSearchCV(
                pipeline, param_distributions=param_dist, n_iter=5, cv=tscv,
                scoring='neg_mean_squared_error', n_jobs=3, random_state=42, verbose=1
            )
            
            # Fit the model
            search.fit(X_train_val, Y_train_val)
            best_model = search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics (same as RF)
            mae = mean_absolute_error(Y_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test, y_pred)
            mape = np.mean(np.abs((Y_test - y_pred) / np.maximum(Y_test, 1e-10))) * 100
            
            # Log results
            logger.info(f"XGB Volatility Model - MAE: {mae}")
            logger.info(f"XGB Volatility Model - RMSE: {rmse}")
            logger.info(f"XGB Volatility Model - R² Score: {r2}")
            logger.info(f"XGB Volatility Model - MAPE: {mape:.2f}%")
            
            # Save model, scaler, and metrics
            model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_xgb_model_{prediction_horizon}m.pkl')
            joblib.dump(best_model, model_path)
            
            scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_scaler_{prediction_horizon}m.pkl')
            joblib.dump(scaler, scaler_path)
            
            metrics = {
                'model': f'{model_type}_volatility',
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
            
            metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_xgb_metrics_{prediction_horizon}m.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            logger.info(f"XGB Volatility Model for {token_name} saved to {model_path}")
            
            return metrics
            
        elif model_type == 'lstm':
            # LSTM for volatility prediction
            X_train, X_val = X_train_val[:int(0.8*len(X_train_val))], X_train_val[int(0.8*len(X_train_val)):]
            Y_train, Y_val = Y_train_val[:int(0.8*len(Y_train_val))], Y_train_val[int(0.8*len(Y_train_val)):]
            
            # Try a simple LSTM architecture for faster training
            model = Sequential()
            model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=30, return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            
            # Compile model with Adam optimizer
            optimizer = Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            # Callbacks for training
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            
            # Train the model with reduced epochs
            history = model.fit(
                X_train, Y_train,
                epochs=50,  # Reduced epochs for faster training
                batch_size=32,
                validation_data=(X_val, Y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(Y_test, y_pred)
            mse = mean_squared_error(Y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(Y_test, y_pred)
            mape = np.mean(np.abs((Y_test - y_pred.flatten()) / np.maximum(Y_test, 1e-10))) * 100
            
            # Log results
            logger.info(f"LSTM Volatility Model - MAE: {mae}")
            logger.info(f"LSTM Volatility Model - RMSE: {rmse}")
            logger.info(f"LSTM Volatility Model - R² Score: {r2}")
            logger.info(f"LSTM Volatility Model - MAPE: {mape:.2f}%")
            
            # Plot training & validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title(f'LSTM Volatility Model Loss for {token_name} ({prediction_horizon}m)')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper right')
            plt.savefig(os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_lstm_loss_{prediction_horizon}m.png'))
            
            # Save model, scaler, and metrics
            model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_model_{prediction_horizon}m.keras')
            save_model(model, model_path)
            
            scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_scaler_{prediction_horizon}m.pkl')
            joblib.dump(scaler, scaler_path)
            
            metrics = {
                'model': f'{model_type}_volatility',
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape)
            }
            
            metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_lstm_metrics_{prediction_horizon}m.json')
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
                
            logger.info(f"LSTM Volatility Model for {token_name} saved to {model_path}")
            
            return metrics
            
    except Exception as e:
        logger.error(f"Error training volatility model for {token_name}: {e}", exc_info=True)
        return None

# Define different time horizons for model training
time_horizons = {
    '60m': (60, 60),      # 1-hour prediction with 60 data points lookback for BERA log-return
    '360m': (72, 360),    # 6-hour prediction with 72 data points lookback for ETH volatility
}

# Define how much historical data to use (in hours)
data_requirements = {
    '60m': 720,       # 30 days * 24 hours for BERA log-return
    '360m': 2160,     # 90 days * 24 hours for ETH volatility
}

# Main execution
if __name__ == "__main__":
    logger.info("Starting model training process")
    
    # Check if necessary directories exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created models directory at {MODELS_DIR}")
    
    # Download data for new tokens if needed
    logger.info("Checking/downloading data for BERA")
    download_token_data("bera", interval="1m", months=2)
    
    # Train models for BERA/USD Log-Return with 1-hour horizon
    token_name = "berausd"
    horizon_name = "60m"
    look_back, prediction_horizon = time_horizons[horizon_name]
    hours_data = data_requirements[horizon_name]
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Log-Return models for {token_name.upper()} with {prediction_horizon}-minute horizon")
    logger.info(f"{'='*50}")
    
    # Train all three model types for BERA log-return
    bera_results = {}
    
    logger.info(f"Starting RandomForest training for {token_name} Log-Return")
    rf_metrics = train_and_save_log_return_model(token_name, look_back, prediction_horizon, hours_data, 'rf')
    if rf_metrics:
        bera_results['rf'] = rf_metrics
    
    logger.info(f"Starting XGBoost training for {token_name} Log-Return")
    xgb_metrics = train_and_save_log_return_model(token_name, look_back, prediction_horizon, hours_data, 'xgb')
    if xgb_metrics:
        bera_results['xgb'] = xgb_metrics
    
    logger.info(f"Starting LSTM training for {token_name} Log-Return")
    lstm_metrics = train_and_save_log_return_model(token_name, look_back, prediction_horizon, hours_data, 'lstm')
    if lstm_metrics:
        bera_results['lstm'] = lstm_metrics
    
    # Save comparison for BERA log-return
    if bera_results:
        comparison_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_comparison_{prediction_horizon}m.json')
        with open(comparison_path, 'w') as f:
            json.dump(bera_results, f)
        
        # Determine best model by MZTAE (lower is better)
        best_bera_model = min(bera_results.items(), key=lambda x: x[1]['mztae'])[0]
        logger.info(f"Best model for {token_name} log-return: {best_bera_model} (MZTAE: {bera_results[best_bera_model]['mztae']})")
        
        # Create comparison plots for BERA Log-Return
        plt.figure(figsize=(15, 10))
        
        # MZTAE comparison (lower is better)
        plt.subplot(2, 2, 1)
        models = list(bera_results.keys())
        mztae_values = [bera_results[model]['mztae'] for model in models]
        plt.bar(models, mztae_values)
        plt.title('MZTAE Comparison (Lower is Better)')
        plt.ylabel('MZTAE')
        
        # MAE comparison (lower is better)
        plt.subplot(2, 2, 2)
        mae_values = [bera_results[model]['mae'] for model in models]
        plt.bar(models, mae_values)
        plt.title('MAE Comparison (Lower is Better)')
        plt.ylabel('MAE')
        
        # R² comparison (higher is better)
        plt.subplot(2, 2, 3)
        r2_values = [bera_results[model]['r2'] for model in models]
        plt.bar(models, r2_values)
        plt.title('R² Comparison (Higher is Better)')
        plt.ylabel('R²')
        
        # Directional accuracy comparison (higher is better)
        plt.subplot(2, 2, 4)
        dir_acc_values = [bera_results[model]['directional_accuracy'] for model in models]
        plt.bar(models, dir_acc_values)
        plt.title('Directional Accuracy Comparison (Higher is Better)')
        plt.ylabel('Directional Accuracy (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_comparison_{prediction_horizon}m.png'))
    
    # Train models for ETH/USD volatility prediction with 6-hour horizon
    token_name = "ethusd"
    horizon_name = "360m"
    look_back, prediction_horizon = time_horizons[horizon_name]
    hours_data = data_requirements[horizon_name]
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Volatility Prediction models for {token_name.upper()} with {prediction_horizon}-minute horizon")
    logger.info(f"{'='*50}")
    
    # Train all three model types for ETH volatility
    eth_results = {}
    
    logger.info(f"Starting RandomForest training for {token_name} Volatility")
    rf_metrics = train_and_save_volatility_model(token_name, look_back, prediction_horizon, hours_data, 'rf')
    if rf_metrics:
        eth_results['rf'] = rf_metrics
    
    logger.info(f"Starting XGBoost training for {token_name} Volatility")
    xgb_metrics = train_and_save_volatility_model(token_name, look_back, prediction_horizon, hours_data, 'xgb')
    if xgb_metrics:
        eth_results['xgb'] = xgb_metrics
    
    logger.info(f"Starting LSTM training for {token_name} Volatility")
    lstm_metrics = train_and_save_volatility_model(token_name, look_back, prediction_horizon, hours_data, 'lstm')
    if lstm_metrics:
        eth_results['lstm'] = lstm_metrics
    
    # Save comparison for ETH volatility
    if eth_results:
        comparison_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_comparison_{prediction_horizon}m.json')
        with open(comparison_path, 'w') as f:
            json.dump(eth_results, f)
        
        # Determine best model by RMSE (lower is better)
        best_eth_model = min(eth_results.items(), key=lambda x: x[1]['rmse'])[0]
        logger.info(f"Best model for {token_name} volatility: {best_eth_model} (RMSE: {eth_results[best_eth_model]['rmse']})")
        
        # Create comparison plots for ETH volatility
        plt.figure(figsize=(15, 10))
        
        # RMSE comparison (lower is better)
        plt.subplot(2, 2, 1)
        models = list(eth_results.keys())
        rmse_values = [eth_results[model]['rmse'] for model in models]
        plt.bar(models, rmse_values)
        plt.title('RMSE Comparison (Lower is Better)')
        plt.ylabel('RMSE')
        
        # MAE comparison (lower is better)
        plt.subplot(2, 2, 2)
        mae_values = [eth_results[model]['mae'] for model in models]
        plt.bar(models, mae_values)
        plt.title('MAE Comparison (Lower is Better)')
        plt.ylabel('MAE')
        
        # R² comparison (higher is better)
        plt.subplot(2, 2, 3)
        r2_values = [eth_results[model]['r2'] for model in models]
        plt.bar(models, r2_values)
        plt.title('R² Comparison (Higher is Better)')
        plt.ylabel('R²')
        
        # MAPE comparison (lower is better)
        plt.subplot(2, 2, 4)
        mape_values = [eth_results[model]['mape'] for model in models]
        plt.bar(models, mape_values)
        plt.title('MAPE Comparison (Lower is Better)')
        plt.ylabel('MAPE (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f'{token_name.lower()}_volatility_comparison_{prediction_horizon}m.png'))
    
    # Print summary of all models
    logger.info("\nSummary of Models:")
    
    if bera_results:
        logger.info(f"BERA/USD Log-Return (60m): {best_bera_model}")
        logger.info(f"  MZTAE: {bera_results[best_bera_model]['mztae']}")
        logger.info(f"  Directional Accuracy: {bera_results[best_bera_model]['directional_accuracy']}%")
    else:
        logger.info("BERA/USD Log-Return (60m): No model trained")
    
    if eth_results:
        logger.info(f"ETH/USD Volatility (360m): {best_eth_model}")
        logger.info(f"  RMSE: {eth_results[best_eth_model]['rmse']}")
        logger.info(f"  MAPE: {eth_results[best_eth_model]['mape']}%")
    else:
        logger.info("ETH/USD Volatility (360m): No model trained")
    
    logger.info("Training process completed!")