import numpy as np
import sqlite3
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import logging
import json
import time
import traceback
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'bera_model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Import app_config for paths
from app_config import DATABASE_PATH, DATA_BASE_PATH, TIINGO_CACHE_DIR, OKX_CACHE_DIR

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Function to load data from Tiingo cache
def load_data_from_tiingo_cache(token_name, cache_dir=TIINGO_CACHE_DIR):
    """
    Memuat data OHLCV dari file cache Tiingo JSON
    
    Args:
        token_name (str): Nama token (misalnya 'berausd')
        cache_dir (str): Direktori cache Tiingo
        
    Returns:
        pd.DataFrame: DataFrame dengan data OHLCV
    """
    # Pastikan token_name tidak memiliki duplikat 'usd'
    base_ticker = token_name.replace('usd', '').lower()
    cache_path = os.path.join(cache_dir, f"{base_ticker}usd_1min_cache.json")
    
    if not os.path.exists(cache_path):
        logger.error(f"Cache file not found: {cache_path}")
        return None
        
    try:
        # Baca file JSON
        with open(cache_path, 'r') as f:
            data = json.load(f)
            
        # Konversi ke DataFrame
        df = pd.DataFrame(data)
        
        # Standardisasi nama kolom
        df.rename(columns={
            'date': 'timestamp',
            'close': 'price'
        }, inplace=True)
        
        # Konversi timestamp ke format UNIX jika belum
        if 'timestamp' in df.columns and not pd.api.types.is_numeric_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) // 10**9
        
        # Memastikan data diurutkan berdasarkan timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Pastikan semua kolom yang diperlukan ada
        required_columns = ['price', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'price' and 'close' in df.columns:
                    df['price'] = df['close']
                elif col != 'price' and 'price' in df.columns:
                    df[col] = df['price']
                elif col == 'volume' and col not in df.columns:
                    df[col] = 0.0
        
        logger.info(f"Loaded {len(df)} records from Tiingo cache for {token_name}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading Tiingo cache data: {e}")
        logger.error(traceback.format_exc())
        return None

# Function to load data from OKX cache
def load_data_from_okx_cache(token_name, cache_dir=OKX_CACHE_DIR):
    """
    Memuat data OHLCV dari file cache OKX CSV
    
    Args:
        token_name (str): Nama token (misalnya 'berausd')
        cache_dir (str): Direktori cache OKX
        
    Returns:
        pd.DataFrame: DataFrame dengan data OHLCV
    """
    # Format nama file cache OKX
    symbol = "BERA_USDT"  # Default untuk BERA
    if token_name.lower() != 'berausd':
        # Jika bukan BERA, format sesuai dengan token
        symbol = f"{token_name.replace('usd', '').upper()}_USDT"
    
    # Cari file cache yang paling baru
    current_date = datetime.now()
    dates_to_try = [
        current_date,
        current_date - timedelta(days=1),
        current_date - timedelta(days=2)
    ]
    
    cache_path = None
    for date in dates_to_try:
        date_str = date.strftime("%Y%m%d")
        possible_path = os.path.join(cache_dir, f"{symbol}_1m_{date_str}.csv")
        if os.path.exists(possible_path):
            cache_path = possible_path
            break
    
    if cache_path is None:
        logger.error(f"OKX cache file not found for {token_name}")
        return None
        
    try:
        # Baca file CSV
        df = pd.read_csv(cache_path)
        
        logger.info(f"Loaded data from OKX cache file: {cache_path}")
        
        # Memastikan data diurutkan berdasarkan timestamp
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Pastikan semua kolom yang diperlukan ada
        required_columns = ['price', 'open', 'high', 'low', 'volume']
        for col in required_columns:
            if col not in df.columns:
                if col == 'price' and 'close' in df.columns:
                    df['price'] = df['close']
                elif col != 'price' and 'price' in df.columns:
                    df[col] = df['price']
                elif col == 'volume' and col not in df.columns:
                    df[col] = 0.0
        
        logger.info(f"Loaded {len(df)} records from OKX cache for {token_name}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading OKX cache data: {e}")
        logger.error(traceback.format_exc())
        return None

# Fetch data from the database (keeping for compatibility)
def load_data_from_db(token_name, hours_back=None):
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if OHLCV columns exist
            cursor.execute("PRAGMA table_info(prices)")
            columns = [col[1] for col in cursor.fetchall()]
            has_ohlcv = all(col in columns for col in ['open', 'high', 'low', 'volume'])
            
            if has_ohlcv:
                # Use OHLCV format
                if hours_back:
                    query = """
                        SELECT price, open, high, low, volume, timestamp FROM prices 
                        WHERE token=? AND timestamp >= (SELECT MAX(timestamp) FROM prices WHERE token=?) - ? * 3600
                        ORDER BY block_height ASC
                    """
                    cursor.execute(query, (token_name.lower(), token_name.lower(), hours_back))
                else:
                    query = """
                        SELECT price, open, high, low, volume, timestamp FROM prices 
                        WHERE token=?
                        ORDER BY block_height ASC
                    """
                    cursor.execute(query, (token_name.lower(),))
                    
                result = cursor.fetchall()
                
                if result:
                    # Create DataFrame with OHLCV data
                    df = pd.DataFrame(result, columns=['price', 'open', 'high', 'low', 'volume', 'timestamp'])
                    logger.info(f"Loaded {len(df)} OHLCV data points from database for {token_name}")
                    return df
            else:
                # Original query with just price
                if hours_back:
                    query = """
                        SELECT price, timestamp FROM prices 
                        WHERE token=? AND timestamp IS NOT NULL AND timestamp >= (SELECT MAX(timestamp) FROM prices WHERE token=?) - ? * 3600
                        ORDER BY block_height ASC
                    """
                    cursor.execute(query, (token_name.lower(), token_name.lower(), hours_back))
                else:
                    query = """
                        SELECT price, timestamp FROM prices 
                        WHERE token=? AND timestamp IS NOT NULL
                        ORDER BY block_height ASC
                    """
                    cursor.execute(query, (token_name.lower(),))
                    
                result = cursor.fetchall()
                
                if result:
                    # Create DataFrame with price and timestamp
                    df = pd.DataFrame(result, columns=['price', 'timestamp'])
                    logger.info(f"Loaded {len(df)} price data points from database for {token_name}")
                    return df
        
        logger.warning(f"No data found in database for {token_name}")
        return None
    
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        logger.error(traceback.format_exc())
        return None

# Combined data loading function - prioritizing Tiingo and OKX
def load_data(token_name, hours_back=None):
    """
    Load data for the specified token, prioritizing Tiingo and OKX cache
    
    Args:
        token_name (str): Token name (e.g., 'berausd')
        hours_back (int): Hours of data to load
        
    Returns:
        pandas.DataFrame: DataFrame with OHLCV data
    """
    # Try loading from Tiingo cache first
    logger.info(f"Trying to load data from Tiingo cache for {token_name}")
    df_tiingo = load_data_from_tiingo_cache(token_name)
    
    # Try loading from OKX cache
    logger.info(f"Trying to load data from OKX cache for {token_name}")
    df_okx = load_data_from_okx_cache(token_name)
    
    # Combine data if both sources available
    if df_tiingo is not None and df_okx is not None:
        # Use the one with more data
        if len(df_tiingo) >= len(df_okx):
            logger.info(f"Using Tiingo data with {len(df_tiingo)} records (more than OKX's {len(df_okx)})")
            df = df_tiingo
        else:
            logger.info(f"Using OKX data with {len(df_okx)} records (more than Tiingo's {len(df_tiingo)})")
            df = df_okx
    # If only one source is available, use it
    elif df_tiingo is not None:
        logger.info(f"Using Tiingo data with {len(df_tiingo)} records (OKX data not available)")
        df = df_tiingo
    elif df_okx is not None:
        logger.info(f"Using OKX data with {len(df_okx)} records (Tiingo data not available)")
        df = df_okx
    # If neither source available, try database as fallback
    else:
        logger.info(f"Data not found in Tiingo or OKX cache for {token_name}, trying database...")
        df = load_data_from_db(token_name, hours_back)
    
    if df is None:
        raise ValueError(f"Could not load data for {token_name} from any source")
    
    # If hours_back is specified and we have timestamps, filter the data
    if hours_back is not None and 'timestamp' in df.columns:
        max_timestamp = df['timestamp'].max()
        cutoff_timestamp = max_timestamp - (hours_back * 3600)
        df = df[df['timestamp'] >= cutoff_timestamp]
        logger.info(f"Filtered data to last {hours_back} hours: {len(df)} rows")
    
    logger.info(f"Loaded {len(df)} data points for {token_name}")
    return df

#-----------------------------------------------------------------------
# Functions for BERA Log-Return Prediction
#-----------------------------------------------------------------------

def calculate_log_returns(prices, period=60):
    """
    Calculate log returns for the given prices with specified period.
    
    Args:
        prices (numpy.array or pandas.Series): Price data
        period (int): Period for calculating returns in data points (default: 60 for 1 hour with 1-minute data)
        
    Returns:
        numpy.array: Array of log returns
    """
    # Ensure prices is a flattened array
    if isinstance(prices, np.ndarray):
        if prices.ndim > 1:
            prices = prices.flatten()
    elif isinstance(prices, pd.Series):
        prices = prices.values
    else:
        # Convert to numpy array if it's not already
        prices = np.array(prices)
    
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

def prepare_features_for_prediction(df, look_back, scaler=None):
    """
    Menyiapkan fitur untuk prediksi model log-return dengan 17 fitur standar
    
    Args:
        df (DataFrame): Data OHLCV yang sudah diurutkan berdasarkan timestamp
        look_back (int): Jumlah data points untuk window
        scaler (MinMaxScaler, optional): Scaler yang sudah di-fit pada data training
        
    Returns:
        numpy.array: Array fitur yang telah di-scale (17 fitur)
    """
    try:
        logger.info(f"Menyiapkan fitur prediksi dari {len(df)} titik data")
        
        # Pastikan semua kolom yang diperlukan ada
        required_cols = ['price', 'open', 'high', 'low']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Kolom yang diperlukan tidak ada: {missing_cols}")
            if 'price' in df.columns:
                for col in missing_cols:
                    if col != 'price':
                        df[col] = df['price']
            else:
                logger.error("Tidak dapat menyiapkan fitur: kolom 'price' tidak ada")
                return None
        
        # Pastikan data cukup untuk window terbesar (20 untuk SMA)
        if len(df) < 20:
            logger.error(f"Data tidak cukup untuk persiapan fitur: dibutuhkan minimal 20, didapat {len(df)}")
            return None
        
        # Persiapkan fitur teknikal
        # 1. Price range
        df['price_range'] = df['high'] - df['low']
        
        # 2. Body size
        df['body_size'] = abs(df['price'] - df['open'])
        
        # 3. Price position
        df['price_position'] = (df['price'] - df['low']) / df['price_range'].replace(0, 1)
        
        # 4. Normalized volume
        if 'volume' in df.columns:
            df['norm_volume'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
        else:
            df['norm_volume'] = 0.0
        
        # 5. Simple Moving Averages
        df['sma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
        df['sma_10'] = df['price'].rolling(window=10, min_periods=1).mean()
        df['sma_20'] = df['price'].rolling(window=20, min_periods=1).mean()
        
        # 6. Distance from SMAs
        df['dist_sma_5'] = (df['price'] - df['sma_5']) / df['price']
        df['dist_sma_10'] = (df['price'] - df['sma_10']) / df['price']
        df['dist_sma_20'] = (df['price'] - df['sma_20']) / df['price']
        
        # 7. Momentum
        df['momentum_5'] = df['price'].pct_change(periods=5)
        df['momentum_10'] = df['price'].pct_change(periods=10)
        df['momentum_20'] = df['price'].pct_change(periods=20)
        
        # Isi NaN values
        df.fillna(0, inplace=True)
        
        # Daftar fitur yang sama seperti saat training
        feature_columns = [
            'price', 'open', 'high', 'low', 'price_range',
            'body_size', 'price_position', 'norm_volume',
            'sma_5', 'dist_sma_5', 'sma_10', 'dist_sma_10',
            'sma_20', 'dist_sma_20', 'momentum_5',
            'momentum_10', 'momentum_20'
        ]
        
        # Ambil look_back baris terakhir untuk window
        if len(df) >= look_back:
            feature_window = df.iloc[-look_back:][feature_columns].values
            
            # Flatten window jika diperlukan
            flattened_features = feature_window.flatten()
            features = flattened_features.reshape(1, -1)
            
            # Terapkan scaling jika scaler disediakan
            if scaler is not None:
                features = scaler.transform(features)
            
            logger.info(f"Berhasil menyiapkan fitur prediksi dengan shape: {features.shape}")
            return features
        else:
            logger.error(f"Tidak cukup data untuk window size {look_back}")
            return None
        
    except Exception as e:
        logger.error(f"Error menyiapkan fitur prediksi: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def prepare_data_for_log_return(data_df, look_back, prediction_period):
    """
    Prepare data for training log-return prediction models.
    Uses a dataframe with OHLCV data for enhanced feature engineering.
    
    Args:
        data_df (pandas.DataFrame): DataFrame with price and other OHLCV data
        look_back (int): Number of data points to use for prediction
        prediction_period (int): Prediction horizon in data points
        
    Returns:
        X, Y, prev_Y, scaler: Prepared training data and scaler
    """
    logger.info(f"Preparing data with look_back={look_back}, prediction_period={prediction_period}")
    
    # Check input is a DataFrame
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Ensure all required columns exist
    required_cols = ['price']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    
    # Create a copy to avoid modifying the original DataFrame
    df = data_df.copy()
    
    # Calculate log returns for target variable
    price_array = df['price'].values
    
    # Ensure we have enough data for the prediction period
    if len(price_array) <= prediction_period:
        logger.error(f"Not enough data points ({len(price_array)}) for prediction period ({prediction_period})")
        return None, None, None, None
        
    all_log_returns = calculate_log_returns(price_array, prediction_period)
    
    # Create feature list, starting with price
    features_list = ['price']
    
    # Add OHLC columns if available
    if all(col in df.columns for col in ['open', 'high', 'low']):
        features_list.extend(['open', 'high', 'low'])
        
        # Calculate price movement features
        df['price_range'] = (df['high'] - df['low']) / df['open']  # Normalized price range
        df['body_size'] = abs(df['price'] - df['open']) / df['open']  # Normalized body size
        features_list.extend(['price_range', 'body_size'])
        
        # Price position within range
        df['price_position'] = (df['price'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        features_list.append('price_position')
    
    # Add volume if available
    if 'volume' in df.columns:
        # Normalize volume by dividing by its rolling mean
        df['norm_volume'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
        features_list.append('norm_volume')
    
    # Add technical indicators as features
    # Simple Moving Averages
    if len(df) > 20:
        for window in [5, 10, 20]:
            col_name = f'sma_{window}'
            df[col_name] = df['price'].rolling(window=window, min_periods=1).mean()
            # Calculate distance from moving average
            df[f'dist_sma_{window}'] = (df['price'] - df[col_name]) / df[col_name]
            features_list.extend([col_name, f'dist_sma_{window}'])
    
    # Calculate momentum (price change over periods)
    for period in [5, 10, 20]:
        if len(df) > period:
            col_name = f'momentum_{period}'
            df[col_name] = df['price'].pct_change(periods=period)
            features_list.append(col_name)
    
    # Fill NaN values with appropriate method
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Adjust data range to align with log returns
    # We need to remove prediction_period data points from the end when calculating features
    df_adjusted = df.iloc[:-prediction_period].copy()
    
    # Debug log for data shapes
    logger.info(f"Original dataframe shape: {df.shape}")
    logger.info(f"Adjusted dataframe shape: {df_adjusted.shape}")
    logger.info(f"Log returns shape: {all_log_returns.shape}")
    
    # Scale the features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    if len(df_adjusted) > 0:
        # Set feature names to ensure consistency
        feature_scaler.feature_names_in_ = np.array(features_list)
        scaled_features = feature_scaler.fit_transform(df_adjusted[features_list])
    else:
        logger.error("No data available after adjustments")
        return None, None, None, None
    
    # Prepare samples for XGBoost/RF model (flattened windows)
    X, Y, prev_Y = [], [], []
    
    # For each possible window
    for i in range(len(scaled_features) - look_back + 1):
        if i < len(all_log_returns):
            # For tree-based models: flatten the window into a single vector
            window_features = scaled_features[i:i + look_back].flatten()
            X.append(window_features)
            
            # Y is the log return for the prediction period
            Y.append(all_log_returns[i])
            
            # prev_Y is the last price in the window (for directional accuracy)
            prev_Y.append(scaled_features[i + look_back - 1][0])  # price is at index 0
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    
    # Debug log for data shapes
    logger.info(f"X shape after preparation: {X.shape}")
    logger.info(f"Y shape after preparation: {Y.shape}")
    logger.info(f"prev_Y shape after preparation: {prev_Y.shape}")
    
    # Verify that X has the expected shape (samples, 1020) for 60 timesteps and 17 features
    expected_features = look_back * len(features_list)
    if X.shape[1] != expected_features:
        logger.warning(f"X shape {X.shape[1]} doesn't match expected features {expected_features}. Training may fail.")
    
    return X, Y, prev_Y, feature_scaler

def prepare_data_for_lstm(data_df, look_back, prediction_period):
    """
    Prepare data for LSTM model with proper 3D shape
    """
    logger.info(f"Preparing data for LSTM with look_back={look_back}, prediction_period={prediction_period}")
    
    # Check input is a DataFrame
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    
    # Ensure all required columns exist
    required_cols = ['price']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Required columns missing: {missing_cols}")
    
    # Create a copy to avoid modifying the original DataFrame
    df = data_df.copy()
    
    # Calculate log returns for target variable
    price_array = df['price'].values
    
    # Ensure we have enough data for the prediction period
    if len(price_array) <= prediction_period:
        logger.error(f"Not enough data points ({len(price_array)}) for prediction period ({prediction_period})")
        return None, None, None, None
        
    all_log_returns = calculate_log_returns(price_array, prediction_period)
    
    # Create feature list, starting with price
    features_list = ['price']
    
    # Add OHLC columns if available
    if all(col in df.columns for col in ['open', 'high', 'low']):
        features_list.extend(['open', 'high', 'low'])
        
        # Calculate price movement features
        df['price_range'] = (df['high'] - df['low']) / df['open']  # Normalized price range
        df['body_size'] = abs(df['price'] - df['open']) / df['open']  # Normalized body size
        features_list.extend(['price_range', 'body_size'])
        
        # Price position within range
        df['price_position'] = (df['price'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        features_list.append('price_position')
    
    # Add volume if available
    if 'volume' in df.columns:
        # Normalize volume by dividing by its rolling mean
        df['norm_volume'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
        features_list.append('norm_volume')
    
    # Add technical indicators as features
    # Simple Moving Averages
    if len(df) > 20:
        for window in [5, 10, 20]:
            col_name = f'sma_{window}'
            df[col_name] = df['price'].rolling(window=window, min_periods=1).mean()
            # Calculate distance from moving average
            df[f'dist_sma_{window}'] = (df['price'] - df[col_name]) / df[col_name]
            features_list.extend([col_name, f'dist_sma_{window}'])
    
    # Calculate momentum (price change over periods)
    for period in [5, 10, 20]:
        if len(df) > period:
            col_name = f'momentum_{period}'
            df[col_name] = df['price'].pct_change(periods=period)
            features_list.append(col_name)
    
    # Fill NaN values with appropriate method
    df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Adjust data range to align with log returns
    # We need to remove prediction_period data points from the end when calculating features
    df_adjusted = df.iloc[:-prediction_period].copy()
    
    # Debug log for data shapes
    logger.info(f"Original dataframe shape for LSTM: {df.shape}")
    logger.info(f"Adjusted dataframe shape for LSTM: {df_adjusted.shape}")
    
    # Scale the features
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    if len(df_adjusted) > 0:
        # Set feature names to ensure consistency
        feature_scaler.feature_names_in_ = np.array(features_list)
        scaled_features = feature_scaler.fit_transform(df_adjusted[features_list])
    else:
        logger.error("No data available after adjustments for LSTM")
        return None, None, None, None
    
    # Prepare samples for LSTM model (3D shape)
    X, Y, prev_Y = [], [], []
    
    # For each possible window
    for i in range(len(scaled_features) - look_back + 1):
        if i < len(all_log_returns):
            # For LSTM, keep the 2D shape (look_back, features)
            X.append(scaled_features[i:i + look_back])
            
            # Y is the log return for the prediction period
            Y.append(all_log_returns[i])
            
            # prev_Y is the last price in the window (for directional accuracy)
            prev_Y.append(scaled_features[i + look_back - 1][0])  # price is at index 0
    
    # Convert to numpy arrays
    X = np.array(X)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    
    # Debug log for LSTM data shapes
    logger.info(f"LSTM X shape: {X.shape}")
    logger.info(f"LSTM Y shape: {Y.shape}")
    
    return X, Y, prev_Y, feature_scaler

def train_and_save_rf_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon):
    """
    Train and save RandomForest model for log-return prediction
    
    Args:
        token_name (str): Token name (e.g., 'berausd')
        X_train (numpy.array): Training features
        Y_train (numpy.array): Training targets (log returns)
        X_test (numpy.array): Test features
        Y_test (numpy.array): Test targets (log returns)
        prev_Y_test (numpy.array): Previous prices for directional accuracy calculation
        scaler (MinMaxScaler): Fitted scaler for features
        prediction_horizon (int): Prediction horizon in minutes
        
    Returns:
        dict: Metrics for the trained model
    """
    logger.info(f"Training RandomForest model for {token_name}")
    
    # Create TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Setup RandomForest pipeline
    pipeline = Pipeline([('rf', RandomForestRegressor(random_state=42))])
    
    # Hyperparameter grid - reduced for faster training
    param_dist = {
        'rf__n_estimators': [100, 200],
        'rf__max_depth': [10, 20],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2],
        'rf__max_features': ['sqrt', 'log2']
    }
    
    # Use RandomizedSearchCV
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=4, cv=tscv,
        scoring='neg_mean_absolute_error', n_jobs=3, random_state=42, verbose=1, error_score=np.nan
    )
    
    # Fit the model
    search.fit(X_train, Y_train)
    best_model = search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    
    # Calculate MZTAE
    mztae_score = calculate_mztae(Y_test, y_pred)
    
    # Calculate directional accuracy
    dir_acc = 100 * np.mean((Y_test > 0) == (y_pred > 0))
    
    # Log results
    logger.info(f"RF Model - MAE: {mae}")
    logger.info(f"RF Model - RMSE: {rmse}")
    logger.info(f"RF Model - R² Score: {r2}")
    logger.info(f"RF Model - MZTAE Score: {mztae_score}")
    logger.info(f"RF Model - Directional Accuracy: {dir_acc:.2f}%")
    
    # Save model and scaler
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_rf_model_{prediction_horizon}m.pkl')
    joblib.dump(best_model, model_path)
    
    scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_scaler_{prediction_horizon}m.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save metrics
    metrics = {
        'model': 'rf_logreturn',
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
        
    logger.info(f"RF Model for {token_name} saved to {model_path}")
    
    return metrics

def train_and_save_xgb_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon):
    """
    Train and save XGBoost model for log-return prediction
    
    Args:
        token_name (str): Token name (e.g., 'berausd')
        X_train (numpy.array): Training features
        Y_train (numpy.array): Training targets (log returns)
        X_test (numpy.array): Test features
        Y_test (numpy.array): Test targets (log returns)
        prev_Y_test (numpy.array): Previous prices for directional accuracy calculation
        scaler (MinMaxScaler): Fitted scaler for features
        prediction_horizon (int): Prediction horizon in minutes
        
    Returns:
        dict: Metrics for the trained model
    """
    logger.info(f"Training XGBoost model for {token_name}")
    
    # Create TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Setup XGBoost pipeline
    pipeline = Pipeline([('xgb', XGBRegressor(random_state=42, objective='reg:squarederror'))])
    
    # Hyperparameter grid for XGBoost - reduced for faster training
    param_dist = {
        'xgb__n_estimators': [100, 200],
        'xgb__max_depth': [3, 5],
        'xgb__learning_rate': [0.01, 0.05],
        'xgb__subsample': [0.8, 1.0],
        'xgb__colsample_bytree': [0.7, 0.8],
        'xgb__min_child_weight': [1, 2],
        'xgb__gamma': [0, 0.1]
    }
    
    # Use RandomizedSearchCV
    search = RandomizedSearchCV(
        pipeline, param_distributions=param_dist, n_iter=4, cv=tscv,
        scoring='neg_mean_absolute_error', n_jobs=3, random_state=42, verbose=1, error_score=np.nan
    )
    
    # Fit the model
    search.fit(X_train, Y_train)
    best_model = search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    
    # Calculate MZTAE
    mztae_score = calculate_mztae(Y_test, y_pred)
    
    # Calculate directional accuracy
    dir_acc = 100 * np.mean((Y_test > 0) == (y_pred > 0))
    
    # Log results
    logger.info(f"XGB Model - MAE: {mae}")
    logger.info(f"XGB Model - RMSE: {rmse}")
    logger.info(f"XGB Model - R² Score: {r2}")
    logger.info(f"XGB Model - MZTAE Score: {mztae_score}")
    logger.info(f"XGB Model - Directional Accuracy: {dir_acc:.2f}%")
    
    # Save model and scaler
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_xgb_model_{prediction_horizon}m.pkl')
    joblib.dump(best_model, model_path)
    
    scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_scaler_{prediction_horizon}m.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save metrics
    metrics = {
        'model': 'xgb_logreturn',
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
        
    logger.info(f"XGB Model for {token_name} saved to {model_path}")
    
    return metrics

def train_and_save_lstm_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon, look_back):
    """
    Train and save LSTM model for log-return prediction
    
    Args:
        token_name (str): Token name (e.g., 'berausd')
        X_train (numpy.array): Training features (3D shape for LSTM)
        Y_train (numpy.array): Training targets (log returns)
        X_test (numpy.array): Test features (3D shape for LSTM)
        Y_test (numpy.array): Test targets (log returns)
        prev_Y_test (numpy.array): Previous prices for directional accuracy calculation
        scaler (MinMaxScaler): Fitted scaler for features
        prediction_horizon (int): Prediction horizon in minutes
        look_back (int): Look back window size
        
    Returns:
        dict: Metrics for the trained model
    """
    logger.info(f"Training LSTM model for {token_name}")
    
    # Split training data into train and validation sets
    X_train_split, X_val = X_train[:int(0.8*len(X_train))], X_train[int(0.8*len(X_train)):]
    Y_train_split, Y_val = Y_train[:int(0.8*len(Y_train))], Y_train[int(0.8*len(Y_train)):]
    
    # Get input dimensions
    n_features = X_train.shape[2]
    
    # Create LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, n_features)))
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
        X_train_split, Y_train_split,
        epochs=35,  # Reduced for faster training
        batch_size=32,
        validation_data=(X_val, Y_val),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Make predictions with smoothing
    y_pred = model.predict(X_test).flatten()
    y_pred = smooth_predictions(y_pred, alpha=0.2)
    
    # Calculate metrics
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    
    # Calculate MZTAE
    mztae_score = calculate_mztae(Y_test, y_pred)
    
    # Calculate directional accuracy
    dir_acc = 100 * np.mean((Y_test > 0) == (y_pred > 0))
    
    # Log results
    logger.info(f"LSTM Model - MAE: {mae}")
    logger.info(f"LSTM Model - RMSE: {rmse}")
    logger.info(f"LSTM Model - R² Score: {r2}")
    logger.info(f"LSTM Model - MZTAE Score: {mztae_score}")
    logger.info(f"LSTM Model - Directional Accuracy: {dir_acc:.2f}%")
    
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
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lstm_model_{prediction_horizon}m.keras')
    save_model(model, model_path)
    
    scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_scaler_{prediction_horizon}m.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save metrics
    metrics = {
        'model': 'lstm_logreturn',
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
        
    logger.info(f"LSTM Model for {token_name} saved to {model_path}")
    
    return metrics

def train_log_return_models(token_name, look_back, prediction_horizon, hours_data=None):
    """
    Wrapper function to train all model types for log-return prediction
    Uses data from Tiingo and OKX cache for better robustness
    
    Args:
        token_name (str): Token name (e.g., 'berausd')
        look_back (int): Number of past data points to use for prediction
        prediction_horizon (int): Prediction horizon in minutes
        hours_data (int): Hours of historical data to use
    
    Returns:
        dict: Results from all models
    """
    try:
        logger.info(f"Training Log-Return models for {token_name} with {prediction_horizon}-minute horizon")
        
        # Load data from the best available source (Tiingo, OKX, or DB)
        data_df = load_data(token_name, hours_data)
        
        # Train test split for all models
        results = {}
        
        # Train Random Forest model
        logger.info("Preparing data for RandomForest model...")
        X, Y, prev_Y, scaler = prepare_data_for_log_return(data_df, look_back, prediction_horizon)
        
        if X is None:
            logger.error("Failed to prepare data for RF model")
            return None
        
        # Split into train and test sets
        test_size = int(0.2 * len(X))
        X_train, X_test = X[:-test_size], X[-test_size:]
        Y_train, Y_test = Y[:-test_size], Y[-test_size:]
        prev_Y_test = prev_Y[-test_size:]
        
        logger.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
        
        # Train RF model
        rf_metrics = train_and_save_rf_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon)
        if rf_metrics:
            results['rf'] = rf_metrics
        
        # Train XGBoost model (uses same data preparation as RF)
        xgb_metrics = train_and_save_xgb_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon)
        if xgb_metrics:
            results['xgb'] = xgb_metrics
        
        # Train LSTM model (needs special data preparation)
        logger.info("Preparing data for LSTM model...")
        X_lstm, Y_lstm, prev_Y_lstm, scaler_lstm = prepare_data_for_lstm(data_df, look_back, prediction_horizon)
        
        if X_lstm is None:
            logger.error("Failed to prepare data for LSTM model")
        else:
            # Split into train and test sets
            test_size_lstm = int(0.2 * len(X_lstm))
            X_train_lstm, X_test_lstm = X_lstm[:-test_size_lstm], X_lstm[-test_size_lstm:]
            Y_train_lstm, Y_test_lstm = Y_lstm[:-test_size_lstm], Y_lstm[-test_size_lstm:]
            prev_Y_test_lstm = prev_Y_lstm[-test_size_lstm:]
            
            logger.info(f"LSTM training set size: {len(X_train_lstm)}, Test set size: {len(X_test_lstm)}")
            
            # Train LSTM model
            lstm_metrics = train_and_save_lstm_model(token_name, X_train_lstm, Y_train_lstm, X_test_lstm, Y_test_lstm, prev_Y_test_lstm, scaler_lstm, prediction_horizon, look_back)
            if lstm_metrics:
                results['lstm'] = lstm_metrics
        
        return results
    
    except Exception as e:
        logger.error(f"Error training log-return models for {token_name}: {e}", exc_info=True)
        return None

# Main execution
if __name__ == "__main__":
    logger.info("Starting model training process")
    
    # Check if necessary directories exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created models directory at {MODELS_DIR}")
    
    # Ensure cache directories exist
    os.makedirs(TIINGO_CACHE_DIR, exist_ok=True)
    os.makedirs(OKX_CACHE_DIR, exist_ok=True)
    logger.info(f"Cache directories: Tiingo={TIINGO_CACHE_DIR}, OKX={OKX_CACHE_DIR}")
    
    # Train models for BERA/USD Log-Return with 1-hour horizon
    token_name = "berausd"
    look_back = 60  # 60 minutes of lookback data
    prediction_horizon = 60  # 1-hour prediction horizon
    hours_data = 720  # 30 days * 24 hours
    
    logger.info(f"\n{'='*50}")
    logger.info(f"Training Log-Return models for {token_name.upper()} with {prediction_horizon}-minute horizon")
    logger.info(f"Look back window: {look_back} minutes, Data history: {hours_data} hours")
    logger.info(f"{'='*50}")
    
    # Train all model types for BERA log-return
    bera_results = train_log_return_models(token_name, look_back, prediction_horizon, hours_data)
    
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
    
    # Print summary of all models
    logger.info("\nSummary of Models:")
    
    if bera_results:
        logger.info(f"BERA/USD Log-Return (60m): {best_bera_model}")
        logger.info(f"  MZTAE: {bera_results[best_bera_model]['mztae']}")
        logger.info(f"  Directional Accuracy: {bera_results[best_bera_model]['directional_accuracy']}%")
    else:
        logger.info("BERA/USD Log-Return (60m): No model trained")
    
    logger.info("Training process completed!")
