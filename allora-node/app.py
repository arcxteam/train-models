import os
import logging
import json
import numpy as np
import joblib
import asyncio
import functools
import pandas as pd
import sqlite3
import traceback
import time
import ccxt
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from datetime import datetime, timedelta
from flask import Flask, Response, jsonify, request
from tensorflow.keras.models import load_model

# Configure logging with more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bera_prediction_api.log')
    ]
)
logger = logging.getLogger(__name__)

# Import app_config for paths and configurations
from app_config import (
    DATABASE_PATH, DATA_BASE_PATH, TIINGO_API_TOKEN, 
    TIINGO_CACHE_DIR, TIINGO_CACHE_TTL, OKX_CACHE_DIR
)

# Import functions from app_utils
from app_utils import (
    get_ohlcv_from_tiingo, get_ohlcv_from_okx, 
    prepare_data_for_lstm, prepare_data_for_log_return
)

app = Flask(__name__)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Override MODELS_DIR if environment variable is set
if os.environ.get('MODELS_DIR'):
    MODELS_DIR = os.environ.get('MODELS_DIR')
    logger.info(f"Using models directory from environment: {MODELS_DIR}")
# Use specified path if it exists and default does not
elif not os.path.exists(MODELS_DIR) and os.path.exists('/root/forge/allora/models/'):
    MODELS_DIR = '/root/forge/allora/models/'
    logger.info(f"Using alternative models directory: {MODELS_DIR}")

# Constants from environment variables
API_PORT = int(os.environ.get('API_PORT', 8000))
LOOK_BACK = int(os.environ.get('LOOK_BACK', 60))  # Default to 60 for BERA log-return model
PREDICTION_HORIZON = int(os.environ.get('PREDICTION_HORIZON', 60))  # Default to 60m for BERA

# Define token info - focused on BERA
TOKEN_INFO = {
    'berausd': {
        'full_name': 'BERA',
        'model_type': 'logreturn',
        'prediction_horizon': 60,   # 60 minutes (1 hour)
        'look_back': 60             # 60 minutes lookback window
    }
}

# HTTP Response Codes
HTTP_RESPONSE_CODE_200 = 200
HTTP_RESPONSE_CODE_404 = 404
HTTP_RESPONSE_CODE_500 = 500

# Directory paths
CSV_DIR = os.path.join(DATA_BASE_PATH, 'binance', 'futures-klines')

# Cache for model selections and predictions
MODEL_SELECTION_CACHE = {}
PREDICTION_CACHE = {}

# Performance tracking dictionary
PREDICTION_PERFORMANCE = {}

def load_model_comparison(token_name, prediction_horizon):
    """Load model comparison data from JSON file"""
    json_path = os.path.join(MODELS_DIR, f"{token_name}_logreturn_comparison_{prediction_horizon}m.json")
    
    try:
        if not os.path.exists(json_path):
            logger.warning(f"Comparison file not found: {json_path}")
            return None
            
        with open(json_path, 'r') as f:
            comparison_data = json.load(f)
        
        logger.info(f"Successfully loaded comparison data from {json_path}")
        return comparison_data
    except Exception as e:
        logger.error(f"Error loading comparison file {json_path}: {e}")
        return None

def get_best_model(comparison_data):
    """
    Get the best model based on MZTAE and directional accuracy
    
    Args:
        comparison_data (dict): Dictionary with model comparison data
        
    Returns:
        str: Best model type ('lstm', 'rf', or 'xgb')
    """
    if not comparison_data:
        logger.warning("No comparison data provided, defaulting to RandomForest")
        return 'rf'  # Default to Random Forest if no comparison data
    
    # Log available models and metrics
    logger.info(f"Available models in comparison data: {list(comparison_data.keys())}")
    
    # For BERA log-return models, we prioritize:
    # 1. MZTAE (lower is better) - primary metric
    # 2. Directional accuracy (higher is better) - secondary metric
    
    # Using MZTAE as primary metric
    best_model = min(comparison_data.keys(), 
                     key=lambda x: comparison_data[x].get('mztae', float('inf')))
    
    # Log the selection metrics
    mztae = comparison_data[best_model].get('mztae', 'N/A')
    dir_acc = comparison_data[best_model].get('directional_accuracy', 'N/A')
    logger.info(f"Selected best model based on MZTAE: {best_model} (MZTAE: {mztae}, Dir Acc: {dir_acc}%)")
    
    return best_model

def determine_best_model(token_name, prediction_horizon):
    """
    Determines the best model for BERA based on metrics in comparison JSON file.
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        prediction_horizon (int): Prediction horizon in minutes
        
    Returns:
        str: Model type ('lstm', 'rf', or 'xgb')
    """
    cache_key = f"{token_name}_{prediction_horizon}"
    
    # Return cached result if available
    if cache_key in MODEL_SELECTION_CACHE:
        logger.info(f"Using cached model selection for {token_name} ({prediction_horizon}m): {MODEL_SELECTION_CACHE[cache_key]}")
        return MODEL_SELECTION_CACHE[cache_key]
    
    # Load comparison data
    comparison_data = load_model_comparison(token_name, prediction_horizon)
    
    if comparison_data:
        best_model = get_best_model(comparison_data)
        
        # Cache the result
        MODEL_SELECTION_CACHE[cache_key] = best_model
        return best_model
    
    # If no comparison file or parsing error, default to Random Forest
    logger.warning(f"No valid comparison data found for {token_name} ({prediction_horizon}m), defaulting to 'rf'")
    MODEL_SELECTION_CACHE[cache_key] = 'rf'
    return 'rf'

def load_model_and_scaler(token_name, prediction_horizon):
    """
    Load the appropriate model and scaler for BERA prediction
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        prediction_horizon (int): Prediction horizon in minutes
        
    Returns:
        tuple: (model, scaler, model_type)
    """
    # Determine the best model to load
    best_model_type = determine_best_model(token_name, prediction_horizon)
    
    # Define file paths based on the selected model type
    if best_model_type == 'lstm':
        model_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_lstm_model_{prediction_horizon}m.keras')
    else:
        model_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_{best_model_type}_model_{prediction_horizon}m.pkl')
    
    scaler_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_scaler_{prediction_horizon}m.pkl')
    
    logger.info(f"Attempting to load {best_model_type} model from: {model_path}")
    logger.info(f"Scaler path: {scaler_path}")
    
    # Attempt to load the best model
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # Load model based on its type
            if best_model_type == 'lstm':
                model = load_model(model_path)
            else:
                model = joblib.load(model_path)
                
            scaler = joblib.load(scaler_path)
            
            # Check scaler attributes for debugging
            if hasattr(scaler, 'n_features_in_'):
                logger.info(f"Scaler expects {scaler.n_features_in_} feature(s) as input")
            if hasattr(scaler, 'feature_names_in_'):
                logger.info(f"Scaler feature names: {scaler.feature_names_in_}")
                
            logger.info(f"Successfully loaded {best_model_type.upper()} model for {token_name}")
            return model, scaler, best_model_type
        except Exception as e:
            logger.error(f"Error loading {best_model_type} model for {token_name}: {e}")
    else:
        logger.warning(f"Best model files not found for {token_name} ({best_model_type})")
    
    # If best model fails, try other models in fallback order
    fallback_order = ['rf', 'xgb', 'lstm'] if best_model_type != 'rf' else ['xgb', 'lstm']
    
    for model_type in fallback_order:
        if model_type == 'lstm':
            model_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_lstm_model_{prediction_horizon}m.keras')
        else:
            model_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_{model_type}_model_{prediction_horizon}m.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                if model_type == 'lstm':
                    model = load_model(model_path)
                else:
                    model = joblib.load(model_path)
                    
                scaler = joblib.load(scaler_path)
                logger.info(f"Loaded fallback {model_type.upper()} model for {token_name}")
                return model, scaler, model_type
            except Exception as e:
                logger.error(f"Error loading fallback {model_type} model for {token_name}: {e}")
    
    logger.error(f"No working models found for {token_name} with {prediction_horizon}m horizon")
    return None, None, None

def load_data_from_db(token_name, limit=None):
    """
    Load OHLCV data from SQLite database
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        limit (int): Number of data points to return
        
    Returns:
        tuple: (prices, block_heights, timestamps, ohlcv_data) or (prices, block_heights, timestamps) for compatibility
    """
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if OHLCV columns exist
            cursor.execute("PRAGMA table_info(prices)")
            columns = [col[1] for col in cursor.fetchall()]
            has_ohlcv = all(col in columns for col in ['open', 'high', 'low', 'volume'])
            
            if has_ohlcv:
                # Use OHLCV format
                query = """
                    SELECT price, block_height, timestamp, open, high, low, volume 
                    FROM prices 
                    WHERE token=?
                    ORDER BY block_height DESC
                """
            else:
                # Use old format
                query = """
                    SELECT price, block_height, timestamp FROM prices 
                    WHERE token=? AND timestamp IS NOT NULL
                    ORDER BY block_height DESC
                """
                if 'timestamp' not in columns:
                    query = """
                        SELECT price, block_height FROM prices 
                        WHERE token=?
                        ORDER BY block_height DESC
                    """
            
            if limit:
                query += " LIMIT ?"
                cursor.execute(query, (token_name, limit))
            else:
                cursor.execute(query, (token_name,))
                
            result = cursor.fetchall()
            
        if result:
            # Convert to numpy array and reverse to get chronological order
            data = np.array(result)
            data = data[::-1]  # Reverse to get chronological order
            
            prices = data[:, 0].astype(float).reshape(-1, 1)
            block_heights = data[:, 1]
            
            if has_ohlcv and data.shape[1] > 6:
                timestamps = data[:, 2]
                
                # Extract OHLCV data
                df = pd.DataFrame({
                    'price': data[:, 0],
                    'timestamp': data[:, 2],
                    'open': data[:, 3],
                    'high': data[:, 4],
                    'low': data[:, 5],
                    'volume': data[:, 6]
                })
                
                logger.info(f"Loaded {len(prices)} OHLCV data points from database for {token_name}")
                return prices, block_heights, timestamps, df
            
            elif data.shape[1] > 2:
                timestamps = data[:, 2]
            else:
                timestamps = np.array([None] * len(prices))
                
            logger.info(f"Loaded {len(prices)} data points from database for {token_name}")
            # For compatibility with old code
            return prices, block_heights, timestamps
        else:
            logger.warning(f"No data found in database for {token_name}")
            return None, None, None
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

def load_data_from_binance(token_name, limit=None):
    """
    Load price data from Binance CSV files
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        limit (int): Number of data points to return
        
    Returns:
        tuple: (prices, block_heights, timestamps)
    """
    # Convert token to proper Binance format
    binance_token = token_name.replace('usd', 'usdt').lower()
    token_dir = os.path.join(CSV_DIR, binance_token)
    
    # Check if combined file exists
    combined_csv_path = os.path.join(token_dir, f"{binance_token}_combined.csv")
    
    if os.path.exists(combined_csv_path):
        try:
            # Load data from combined CSV
            logger.info(f"Loading data from combined CSV: {combined_csv_path}")
            df = pd.read_csv(combined_csv_path)
            
            # Sort by timestamp to ensure chronological order
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Limit data if needed
            if limit:
                df = df.tail(limit)
            
            # Create arrays for output
            prices = df['price'].values.reshape(-1, 1)
            timestamps = df['timestamp'].values
            # Use timestamps as block heights for Binance data
            block_heights = np.array(range(len(prices)))
            
            logger.info(f"Loaded {len(prices)} data points from combined Binance CSV for {token_name}")
            return prices, block_heights, timestamps
            
        except Exception as e:
            logger.error(f"Error loading from combined CSV: {e}")
    
    # If combined file doesn't exist or had errors, try individual CSV files
    if not os.path.exists(token_dir):
        logger.warning(f"Binance data directory not found: {token_dir}")
        return None, None, None
    
    try:
        # Find all CSV files
        csv_files = [f for f in os.listdir(token_dir) if f.endswith('.csv') and not f.endswith('_combined.csv')]
        if not csv_files:
            logger.warning(f"No CSV files found in {token_dir}")
            return None, None, None
        
        # Sort files by date (assuming Binance naming format)
        sorted_files = sorted(csv_files, reverse=True)
        
        # Start with the most recent file and combine data
        all_data = []
        remaining_rows = limit if limit else float('inf')
        
        for csv_file in sorted_files:
            file_path = os.path.join(token_dir, csv_file)
            try:
                # Read CSV (Binance format)
                df = pd.read_csv(file_path, header=None)
                df.columns = ["open_time", "open", "high", "low", "close", 
                              "volume", "close_time", "quote_volume", 
                              "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"]
                
                # Sort by open_time to ensure chronological order
                df = df.sort_values('open_time')
                
                # Take only what we need
                if remaining_rows < len(df):
                    df = df.tail(int(remaining_rows))
                    remaining_rows = 0
                else:
                    remaining_rows -= len(df)
                
                all_data.append(df)
                
                # If we have enough data, stop
                if remaining_rows <= 0:
                    break
            except Exception as e:
                logger.error(f"Error reading Binance CSV {csv_file}: {e}")
        
        if not all_data:
            logger.warning(f"No data loaded from Binance CSVs for {token_name}")
            return None, None, None
        
        # Combine all dataframes
        combined_df = pd.concat(all_data)
        
        # Sort by open_time
        combined_df = combined_df.sort_values('open_time').reset_index(drop=True)
        
        # Convert timestamps from milliseconds to seconds
        combined_df['timestamp'] = combined_df['open_time'] / 1000
        
        # Create arrays for output
        prices = combined_df['close'].values.reshape(-1, 1)
        # Use open_time as block_height for Binance data
        block_heights = combined_df['open_time'].values
        timestamps = combined_df['timestamp'].values
        
        logger.info(f"Loaded {len(prices)} data points from Binance CSVs for {token_name}")
        return prices, block_heights, timestamps
    
    except Exception as e:
        logger.error(f"Error loading Binance data for {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

def load_latest_data_multi_source(token_name, limit=None):
    """
    Load the latest data using multi-source strategy with priority:
    1. Tiingo API
    2. OKX API
    3. Database
    4. Binance CSV
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        limit (int): Number of data points to return
        
    Returns:
        tuple: (prices, block_heights, timestamps, source_used)
    """
    # Variable to track which data source was used
    source_used = "None"
    
    # 1. Tiingo API
    try:
        logger.info(f"Trying to get data from Tiingo API for {token_name}")
        tiingo_data = get_ohlcv_from_tiingo(token_name, resample_freq='1min', days_back=2)
        
        if tiingo_data is not None and len(tiingo_data) >= (limit or 0):
            # Konversi ke array untuk kompatibilitas dengan kode lama
            prices = tiingo_data['price'].values.reshape(-1, 1)
            # Use range as block_heights for Tiingo data
            block_heights = np.array(range(len(prices)))
            timestamps = tiingo_data['timestamp'].values
    
            # Limit data if needed
            if limit and len(prices) > limit:
                tiingo_data = tiingo_data.tail(limit)
                prices = prices[-limit:]
                block_heights = block_heights[-limit:]
                timestamps = timestamps[-limit:]
        
            source_used = "Tiingo API"
            logger.info(f"Successfully loaded {len(prices)} data points from Tiingo API")
            return tiingo_data, block_heights, timestamps, source_used  # Return DataFrame
        else:
            logger.warning("Failed to get sufficient data from Tiingo API")
    except Exception as e:
        logger.warning(f"Error getting data from Tiingo API: {e}")
    
    # 2. OKX API
    try:
        logger.info(f"Trying to get data from OKX API for {token_name}")
        symbol = "BERA/USDT"
        okx_data = get_ohlcv_from_okx(symbol, timeframe='1m', limit=(limit+10 if limit else 100))
        
        if okx_data is not None and len(okx_data) >= (limit or 0):
            # Konversi ke array untuk kompatibilitas dengan kode lama
            prices = okx_data['price'].values.reshape(-1, 1)
            # Use range as block_heights for OKX data
            block_heights = np.array(range(len(prices)))
            timestamps = okx_data['timestamp'].values
            
            # Limit data if needed
            if limit and len(prices) > limit:
                okx_data = okx_data.tail(limit)
                prices = prices[-limit:]
                block_heights = block_heights[-limit:]
                timestamps = timestamps[-limit:]
                
            source_used = "OKX API"
            logger.info(f"Successfully loaded {len(prices)} data points from OKX API")
            return okx_data, block_heights, timestamps, source_used  # Return DataFrame
        else:
            logger.warning("Failed to get sufficient data from OKX API")
    except Exception as e:
        logger.warning(f"Error getting data from OKX API: {e}")
    
    # 3. Database SQLite
    try:
        logger.info(f"Trying to get data from database for {token_name}")
        prices, block_heights, timestamps = load_data_from_db(token_name, limit)
        
        if prices is not None and len(prices) >= (limit or 0):
            source_used = "Database"
            logger.info(f"Successfully loaded {len(prices)} data points from database")
            return prices, block_heights, timestamps, source_used
        else:
            logger.warning("Failed to get sufficient data from database")
    except Exception as e:
        logger.warning(f"Error getting data from database: {e}")
    
    # 4. Binance CSV Files (opsional)
    try:
        logger.info(f"Trying to get data from Binance CSV for {token_name}")
        prices, block_heights, timestamps = load_data_from_binance(token_name, limit)
        
        if prices is not None and len(prices) >= (limit or 0):
            source_used = "Binance CSV"
            logger.info(f"Successfully loaded {len(prices)} data points from Binance CSV")
            return prices, block_heights, timestamps, source_used
        else:
            logger.warning("Failed to get sufficient data from Binance CSV")
    except Exception as e:
        logger.warning(f"Error getting data from Binance CSV: {e}")
    
    # If all sources failed, return None
    logger.error(f"All data sources failed for {token_name}")
    return None, None, None, source_used

def track_prediction(token_name, predicted_price, latest_price, block_height):
    """
    Track prediction for later verification
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        predicted_price (float): Predicted price
        latest_price (float): Current price
        block_height: Block height or timestamp of the prediction
    """
    global PREDICTION_PERFORMANCE
    
    if token_name not in PREDICTION_PERFORMANCE:
        PREDICTION_PERFORMANCE[token_name] = []
    
    # Add new prediction record
    PREDICTION_PERFORMANCE[token_name].append({
        'block_height': block_height,
        'timestamp': time.time(),
        'predicted_price': predicted_price,
        'latest_price': latest_price,
        'prediction_delta': predicted_price - latest_price,
        'prediction_pct': ((predicted_price - latest_price) / latest_price) * 100
    })
    
    # Keep only the latest 100 predictions to avoid memory issues
    if len(PREDICTION_PERFORMANCE[token_name]) > 100:
        PREDICTION_PERFORMANCE[token_name] = PREDICTION_PERFORMANCE[token_name][-100:]

def verify_prediction(token_name, block_height, predicted_price, actual_price):
    """
    Verify prediction against actual price
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        block_height: Block height or timestamp of the prediction
        predicted_price (float): Predicted price
        actual_price (float): Actual price
    """
    global PREDICTION_PERFORMANCE
    
    # Find the prediction record
    for record in PREDICTION_PERFORMANCE.get(token_name, []):
        if record.get('block_height') == block_height:
            # Update with actual data
            record['actual_price'] = actual_price
            record['error'] = abs(predicted_price - actual_price)
            record['error_pct'] = (record['error'] / actual_price) * 100
            record['verified_at'] = time.time()
            
            # Log the verification result
            logger.info(f"Prediction verification for {token_name} at block {block_height}:")
            logger.info(f"  Predicted: ${predicted_price:.4f}, Actual: ${actual_price:.4f}")
            logger.info(f"  Error: ${record['error']:.4f} ({record['error_pct']:.2f}%)")
            
            # If we have at least 5 verified predictions, calculate average error
            verified_records = [r for r in PREDICTION_PERFORMANCE.get(token_name, []) if 'actual_price' in r]
            if len(verified_records) >= 5:
                avg_error = sum(r['error'] for r in verified_records) / len(verified_records)
                avg_error_pct = sum(r['error_pct'] for r in verified_records) / len(verified_records)
                logger.info(f"  Average error for {token_name}: ${avg_error:.4f} ({avg_error_pct:.2f}%)")
            
            break

def get_price_at_block(token_name, block_height):
    """
    Get price at specific block height from database
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        block_height: Block height
        
    Returns:
        float: Price at the specified block height
    """
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT price FROM prices 
                WHERE token=? AND block_height=?
            """, (token_name, block_height))
            result = cursor.fetchone()
        
        if result:
            price = float(result[0])
            # If we have a prediction for this block, verify it
            for records in PREDICTION_PERFORMANCE.values():
                for record in records:
                    if record.get('block_height') == block_height and 'actual_price' not in record:
                        verify_prediction(token_name, block_height, record['predicted_price'], price)
            return price
        else:
            logger.warning(f"No price found for {token_name} at block {block_height}")
            return None
    except Exception as e:
        logger.error(f"Error getting price at block: {e}")
        return None

def calculate_log_returns(prices, period=60):
    """
    Calculate log returns for the given prices with specified period.
    
    Args:
        prices (numpy.array): Array of price data
        period (int): Period for calculating returns in data points (default: 60 for 1 hour)
        
    Returns:
        numpy.array: Array of log returns
    """
    # Ensure prices is a flattened array
    if prices.ndim > 1:
        prices = prices.flatten()
    
    # Calculate log returns: ln(price_t+period / price_t)
    log_returns = np.log(prices[period:] / prices[:-period])
    
    return log_returns

def prepare_prediction_data(prices, look_back, model_type):
    """
    Prepare the latest data for BERA log-return prediction
    
    Args:
        prices (numpy.array): Array of price data
        look_back (int): Number of data points to use for prediction
        model_type (str): Model type ('rf', 'xgb', or 'lstm')
        
    Returns:
        numpy.array: Prepared data for prediction
    """
    if len(prices) < look_back:
        logger.warning(f"Not enough price data for prediction: needed {look_back}, got {len(prices)}")
        return None
    
    # Get the last look_back prices
    prediction_data = prices[-look_back:].flatten()
    logger.info(f"Prepared prediction data with shape: {prediction_data.shape}")
    
    return prediction_data

def prepare_features_for_prediction(df, scaler):
    """
    Menyiapkan fitur untuk prediksi model log-return dengan 17 fitur standar
    
    Args:
        df (DataFrame): Data OHLCV yang sudah diurutkan berdasarkan timestamp
        scaler (MinMaxScaler): Scaler yang sudah di-fit pada data training
        
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
        
        # Persiapkan fitur teknikal (sama seperti di prepare_data_for_log_return)
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
        
        # Ambil baris terakhir (data terbaru)
        latest_features = df.iloc[-1][feature_columns].values.reshape(1, -1)
        
        # Terapkan scaling yang sama dengan training
        scaled_features = scaler.transform(latest_features)
        
        logger.info(f"Berhasil menyiapkan fitur prediksi dengan shape: {scaled_features.shape}")
        return scaled_features
    
    except Exception as e:
        logger.error(f"Error menyiapkan fitur prediksi: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def cached_prediction(token_name, prediction_horizon):
    """
    Generate price prediction with caching for better performance
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        prediction_horizon (int): Prediction horizon in minutes
        
    Returns:
        float: Predicted price
    """
    # Invalidate cache if it's older than 5 minutes
    cache_key = f"{token_name}_{prediction_horizon}"
    if cache_key in PREDICTION_CACHE:
        cache_age = time.time() - PREDICTION_CACHE[cache_key]['timestamp']
        if cache_age < 300:  # 5 minutes in seconds
            logger.info(f"Using cached prediction for {token_name} ({cache_age:.1f}s old)")
            return PREDICTION_CACHE[cache_key]['prediction']
    
    start_time = time.time()
    logger.info(f"Starting prediction for {token_name} (horizon: {prediction_horizon}m)")
    
    # Get token info
    token_config = TOKEN_INFO.get(token_name, {
        'look_back': LOOK_BACK,
        'model_type': 'logreturn',
        'prediction_horizon': prediction_horizon
    })
    
    look_back = token_config.get('look_back', LOOK_BACK)
    
    # Load the selected model and scaler
    model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
    model_load_time = time.time() - start_time
    
    if model is None or scaler is None:
        logger.error(f"Failed to load model for {token_name}")
        return None
    
    logger.info(f"Loaded {model_type.upper()} model in {model_load_time:.3f}s")
    
    # Load the latest price data using multi-source strategy
    data_load_start = time.time()
    prices, block_heights, timestamps, data_source = load_latest_data_multi_source(token_name, look_back)
    data_load_time = time.time() - data_load_start
    
    if prices is None or len(prices) < look_back:
        logger.error(f"Insufficient data for {token_name} prediction: needed {look_back}, got {len(prices) if prices is not None else 0}")
        return None
    
    # Get the latest price and block info for logging
    latest_price = float(prices[-1][0])
    latest_block = int(block_heights[-1]) if block_heights is not None and len(block_heights) > 0 else None
    latest_timestamp = timestamps[-1] if timestamps is not None and len(timestamps) > 0 else None
    
    logger.info(f"Loaded {len(prices)} data points from {data_source} in {data_load_time:.3f}s")
    
    # Preprocess data based on model type
    preprocess_start = time.time()

    if model_type == "lstm":
        # Untuk model LSTM, persiapkan sequence data
        prediction_data = prepare_prediction_data(prices, look_back, model_type)
        if prediction_data is None:
            logger.error(f"Failed to prepare prediction data for {token_name}")
            return None
            
        # Check scaler expected features
        n_scaler_features = getattr(scaler, 'n_features_in_', look_back)
        logger.info(f"Scaler expects {n_scaler_features} features")
        
        # Use last n_scaler_features data points to match scaler
        data_for_scaling = prediction_data[-n_scaler_features:].reshape(1, -1)
        logger.info(f"Data for scaling shape: {data_for_scaling.shape}")
        
        # Scale the data
        scaled_data = scaler.transform(data_for_scaling)
        logger.info(f"LSTM scaled data shape: {scaled_data.shape}")
        
        # Reshape to 3D for LSTM input [samples, time_steps, features]
        if scaled_data.shape[1] != look_back:
            # Special case: handle mismatch
            logger.warning(f"Scaled data has {scaled_data.shape[1]} features but look_back is {look_back}")
            # Try to adapt - this depends on your specific model
            if scaled_data.shape[1] > look_back:
                # If we have more data than needed, take the most recent
                scaled_data = scaled_data[:, -look_back:]
            else:
                # If we have less data than needed, pad with zeros (not ideal but a fallback)
                padded = np.zeros((1, look_back))
                padded[:, -scaled_data.shape[1]:] = scaled_data
                scaled_data = padded
                
        X_pred = scaled_data.reshape(1, look_back, 1)
        logger.info(f"LSTM input data shape: {X_pred.shape}")
        
    else:
        # For tree-based models (RF, XGB), we need to create features
        # Convert price data to DataFrame for feature creation
        df = pd.DataFrame({
            'price': prices.flatten(),
            'timestamp': timestamps
        })
    
        # If we have OHLCV data available, use it
        ohlcv_data = None
        if data_source == "Tiingo API":
            logger.info(f"Using OHLCV data from Tiingo API")
            # Tiingo data is already in DataFrame format with OHLCV
            df = prices  # prices contains dataframe in this case
        elif data_source == "OKX API":
            logger.info(f"Using OHLCV data from OKX API")
            # OKX data is already in DataFrame format with OHLCV
            df = prices  # prices contains dataframe in this case
        elif data_source == "Database" and len(prices) >= look_back:
            try:
                # Try loading OHLCV data from database
                _, _, _, ohlcv_data = load_data_from_db(token_name, limit=look_back+20)  # Extra data for moving averages
                if ohlcv_data is not None:
                    logger.info(f"Using OHLCV data from database for feature preparation")
                    df = ohlcv_data
            except Exception as e:
                logger.warning(f"Error loading OHLCV data from DB: {e}. Using price data only.")
    
        # Ensure we have the required columns
        required_cols = ['price', 'open', 'high', 'low', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}. Filling with price data where possible.")
            if 'price' in df.columns:
                for col in missing_cols:
                    if col != 'price' and col != 'volume':
                        df[col] = df['price'].values
                    elif col == 'volume':
                        df[col] = 0.0
    
        # Prepare features with the new function
        X_pred = prepare_features_for_prediction(df, scaler)
        if X_pred is None:
            logger.error(f"Failed to prepare features for {token_name}")
            return None
        
        # Apply scaling
        X_pred = scaler.transform(X_pred)
        logger.info(f"Tree model scaled data shape: {X_pred.shape}")
        # Jika bentuk data tidak sesuai dengan yang diharapkan model, log peringatan
        expected_features = 1020  # 17 fitur x 60 timesteps
        if X_pred.shape[1] != expected_features:
            logger.warning(f"Feature shape mismatch: model expects {expected_features}, got {X_pred.shape[1]}")
        
    preprocess_time = time.time() - preprocess_start
    
    # Make prediction based on model type
    try:
        predict_start = time.time()
        
        # Now X_pred is already in the correct format for each model type
        # PERBAIKAN: Hanya berikan parameter verbose untuk model LSTM
        if model_type == "lstm":
            pred = model.predict(X_pred, verbose=0)
        else:
            # Untuk model XGB dan RF, jangan gunakan parameter verbose
            pred = model.predict(X_pred)
        
        logger.info(f"Prediction output shape: {pred.shape if hasattr(pred, 'shape') else 'scalar'}")
        
        predict_time = time.time() - predict_start
        
        # For log-return model, pred is the predicted log return
        log_return = pred[0] if hasattr(pred, '__len__') else pred
        
        # Convert log return to actual price: price_t+horizon = price_t * exp(log_return)
        prediction_value = latest_price * np.exp(log_return)
        
        # Ensure it's a scalar
        if hasattr(prediction_value, '__len__'):
            prediction_value = prediction_value[0]
        
        # Calculate prediction direction and percentage
        direction = "UP" if prediction_value > latest_price else "DOWN"
        change_pct = ((prediction_value - latest_price) / latest_price) * 100
        
        # Track the prediction for later verification
        track_prediction(token_name, prediction_value, latest_price, latest_block)
        
        # Calculate total prediction time
        total_prediction_time = time.time() - start_time
        
        # Log comprehensive prediction info
        logger.info(f"=== Prediction for {token_name.upper()} (Model: {model_type.upper()}) ===")
        logger.info(f"  Horizon: {prediction_horizon} minutes")
        logger.info(f"  Block: {latest_block}, Timestamp: {latest_timestamp}")
        logger.info(f"  Latest Price: ${latest_price:.4f}")
        logger.info(f"  Predicted Price: ${prediction_value:.4f}")
        logger.info(f"  Log Return: {log_return:.6f}")
        logger.info(f"  Direction: {direction} ({change_pct:.2f}%)")
        logger.info(f"  Data Source: {data_source}")
        logger.info(f"  Performance: Model Load={model_load_time:.3f}s, Data Load={data_load_time:.3f}s, " +
                    f"Preprocess={preprocess_time:.3f}s, Predict={predict_time:.3f}s, Total={total_prediction_time:.3f}s")
        
        # Update the cache
        PREDICTION_CACHE[cache_key] = {
            'prediction': prediction_value,
            'timestamp': time.time(),
            'log_return': log_return,
            'latest_price': latest_price,
            'change_pct': change_pct,
            'direction': direction,
            'model_type': model_type,
            'data_source': data_source
        }
        
        return prediction_value
    except Exception as e:
        logger.error(f"Error during prediction for {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None

# Endpoint root ("/")
@app.route('/', methods=['GET'])
def home():
    return "Welcome to BERA Price Prediction API", 200

# Endpoint health check ("/health")
@app.route('/health', methods=['GET'])
async def health_check():
    """Simple health check endpoint"""
    return "OK", 200

@app.route('/predict_log_return/<token>', methods=['GET'])
async def predict_log_return(token):
    """
    Endpoint untuk prediksi log-return cryptocurrency
    Menggunakan strategi prioritas multi-sumber data OHLCV
    Digunakan oleh Allora Network sesuai dengan config
    """
    logger.info(f"Menerima permintaan prediksi log-return untuk {token}")
    
    # Hanya mendukung BERA
    if token.upper() != "BERA":
        response = json.dumps({"error": f"Token {token} tidak didukung, hanya BERA yang tersedia"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = "berausd"
    prediction_horizon = TOKEN_INFO[token_name]['prediction_horizon']
    look_back = TOKEN_INFO[token_name]['look_back']
    
    try:
        # Ambil data menggunakan strategi multi-sumber
        prices, block_heights, timestamps, source_used = load_latest_data_multi_source(token_name, look_back)
        
        if prices is None or len(prices) < look_back:
            response = json.dumps({"error": "Tidak cukup data untuk prediksi dari semua sumber"})
            return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
        
        # Tambahkan logging untuk debugging
        logger.info(f"Type of 'prices': {type(prices)}")
        logger.info(f"Shape of 'prices': {prices.shape if hasattr(prices, 'shape') else 'No shape attribute'}")

        # Pastikan 'prices' adalah numpy array sebelum flatten
        df = pd.DataFrame({
            'price': prices.to_numpy().flatten() if isinstance(prices, pd.DataFrame) else prices.flatten(),
            'timestamp': timestamps
        })

        # Jika sumber adalah Database tapi tidak ada timestamp, buat timestamp dummy
        if 'Database' in source_used and timestamps is not None and len(timestamps) > 0 and timestamps[0] is None:
            df['timestamp'] = range(len(prices))
            
        # Ambil harga terkini untuk referensi dan tracking
        current_price = float(df['price'].iloc[-1])
        logger.info(f"Harga terkini BERA: ${current_price:.4f} (Sumber: {source_used})")
        
        # Load model terbaik
        model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
        
        if model is None:
            response = json.dumps({"error": "Model prediksi tidak tersedia"})
            return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')
            
        logger.info(f"Menggunakan model {model_type.upper()} untuk prediksi")
        
        # Siapkan data untuk prediksi sesuai dengan jenis model
        try:
            if model_type == 'lstm':
                # Untuk model LSTM
                X, _, _, _ = prepare_data_for_lstm(df, look_back, prediction_horizon)
        
                if X is None or len(X) == 0:
                    response = json.dumps({"error": "Gagal mempersiapkan data LSTM"})
                    return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')
        
                # Ambil data terbaru untuk prediksi
                X_pred = X[-1:] 
            else:
                # Untuk model tree-based (RF, XGB)
                # Persiapkan fitur menggunakan fungsi baru
                X_pred = prepare_features_for_prediction(df, scaler)
        
                if X_pred is None:
                    response = json.dumps({"error": "Gagal mempersiapkan fitur untuk model"})
                    return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

                # Apply scaling
                X_pred = scaler.transform(X_pred)
                logger.info(f"Scaled data shape for prediction: {X_pred.shape}")

                # Jika bentuk data tidak sesuai dengan yang diharapkan model, log peringatan
                expected_features = 1020  # 17 fitur x 60 timesteps
                if X_pred.shape[1] != expected_features:
                    logger.warning(f"Feature shape mismatch: model expects {expected_features}, got {X_pred.shape[1]}")
                
        except Exception as e:
            logger.error(f"Error menyiapkan data prediksi: {e}")
            logger.error(traceback.format_exc())
            response = json.dumps({"error": f"Gagal mempersiapkan data prediksi: {str(e)}"})
            return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')
        
        # Lakukan prediksi
        logger.info(f"Melakukan prediksi dengan input shape: {X_pred.shape}")
        if model_type == "lstm":
            pred = model.predict(X_pred, verbose=0)
        else:
            pred = model.predict(X_pred)
        
        log_return = pred[0] if hasattr(pred, '__len__') else pred
        
        # Log hasil prediksi
        predicted_price = current_price * np.exp(log_return)
        logger.info(f"Prediksi log-return BERA: {log_return}")
        logger.info(f"Harga saat ini: ${current_price:.4f}, Harga prediksi: ${predicted_price:.4f}")
        
        # Simpan dalam cache untuk referensi
        cache_key = f"{token_name}_{prediction_horizon}"
        PREDICTION_CACHE[cache_key] = {
            'timestamp': time.time(),
            'log_return': float(log_return),
            'latest_price': current_price,
            'prediction': predicted_price,
            'data_source': source_used,
            'model_type': model_type
        }
        
        # Track prediction jika ada block height (untuk verifikasi nanti)
        if source_used == "Database" and block_heights is not None and len(block_heights) > 0:
            latest_block = int(block_heights[-1])
            track_prediction(token_name, predicted_price, current_price, latest_block)
        
        # Kembalikan nilai log-return mentah sebagai teks biasa
        return Response(str(log_return), status=HTTP_RESPONSE_CODE_200, mimetype='text/plain')
        
    except Exception as e:
        logger.error(f"Error memproses prediksi log-return: {e}")
        logger.error(traceback.format_exc())
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/inference/bera', methods=['GET'])
async def get_inference_bera():
    """
    Get price prediction for BERA
    """
    logger.info("Received inference request for BERA")
    
    token_name = "berausd"
    prediction_horizon = TOKEN_INFO[token_name]['prediction_horizon']
    
    try:
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, cached_prediction, token_name, prediction_horizon)

        if prediction is None:
            response = json.dumps({"error": "No data found or model unavailable"})
            return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')

        logger.info(f"BERA inference result: {prediction}")
        return Response(str(prediction), status=HTTP_RESPONSE_CODE_200, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/log_return/bera', methods=['GET'])
async def get_log_return_bera():
    """
    Get detailed log-return prediction for BERA
    """
    logger.info("Received log-return request for BERA")
    
    token_name = "berausd"
    prediction_horizon = TOKEN_INFO[token_name]['prediction_horizon']
    
    try:
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, cached_prediction, token_name, prediction_horizon)

        if prediction is None:
            response = json.dumps({"error": "Failed to calculate log-return prediction"})
            return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

        # Get additional data from cache
        cache_key = f"{token_name}_{prediction_horizon}"
        cache_data = PREDICTION_CACHE.get(cache_key, {})
        
        # Return as JSON
        result = {
            "token": "BERA",
            "log_return": cache_data.get('log_return'),
            "predicted_price": cache_data.get('prediction'),
            "latest_price": cache_data.get('latest_price'),
            "prediction_horizon": f"{prediction_horizon}m",
            "model_type": cache_data.get('model_type', 'unknown').upper(),
            "data_source": cache_data.get('data_source', 'unknown'),
            "timestamp": datetime.fromtimestamp(cache_data.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing log-return request: {e}")
        logger.error(traceback.format_exc())
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/truth/bera/<block_height>', methods=['GET'])
async def get_truth_bera(block_height):
    """
    Endpoint for ground truth data at specific block height
    Used by Allora Network reputers
    """
    logger.info(f"Received truth request for BERA at block {block_height}")
    
    try:
        block_height = int(block_height)
    except ValueError:
        response = json.dumps({"error": "Invalid block height format"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = "berausd"
    
    price = get_price_at_block(token_name, block_height)
    
    if price is None:
        response = json.dumps({"error": "No data found for the specified block height"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    logger.info(f"BERA truth at block {block_height}: {price}")
    return Response(str(price), status=HTTP_RESPONSE_CODE_200, mimetype='text/plain')

@app.route('/details/bera', methods=['GET'])
async def bera_details():
    """Get detailed prediction for BERA"""
    token_name = "berausd"
    prediction_horizon = TOKEN_INFO[token_name]['prediction_horizon']
    
    try:
        # Get the current price and latest data using multi-source strategy
        prices, block_heights, timestamps, data_source = load_latest_data_multi_source(token_name, 1)
        if prices is None or len(prices) == 0:
            return jsonify({"status": "error", "message": "No current price data available"})
            
        current_price = float(prices[0][0])
        current_block = int(block_heights[0]) if block_heights is not None and len(block_heights) > 0 else None
        
        # Get the model type
        model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
        if model is None:
            return jsonify({"status": "error", "message": "No model found for BERA"})
        
        # Get prediction
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, cached_prediction, token_name, prediction_horizon)
        
        if prediction is None:
            return jsonify({"status": "error", "message": "Failed to make prediction"})
        
        # Get data from cache
        cache_key = f"{token_name}_{prediction_horizon}"
        cache_data = PREDICTION_CACHE.get(cache_key, {})
        
        # Calculate change percentage
        change_pct = cache_data.get('change_pct', ((prediction - current_price) / current_price * 100))
        
        # Format prediction horizon
        horizon_str = f"{prediction_horizon}m"
        if prediction_horizon == 60:
            horizon_str = "1h"
            
        # Get prediction accuracy metrics if available
        accuracy_metrics = {}
        verified_records = [r for r in PREDICTION_PERFORMANCE.get(token_name, []) if 'actual_price' in r]
        if verified_records:
            avg_error = sum(r['error'] for r in verified_records) / len(verified_records)
            avg_error_pct = sum(r['error_pct'] for r in verified_records) / len(verified_records)
            
            # Direction accuracy (predicted up/down correctly)
            correct_direction = sum(
                1 for r in verified_records
                if (r['predicted_price'] > r['latest_price'] and r['actual_price'] > r['latest_price']) or
                   (r['predicted_price'] < r['latest_price'] and r['actual_price'] < r['latest_price'])
            )
            direction_accuracy = (correct_direction / len(verified_records)) * 100 if verified_records else 0
            
            accuracy_metrics = {
                'average_error': avg_error,
                'average_error_percent': avg_error_pct,
                'direction_accuracy_percent': direction_accuracy,
                'verified_predictions': len(verified_records)
            }
        
        return jsonify({
            "status": "success",
            "token": "BERA",
            "current_price": current_price,
            "current_block": current_block,
            "predicted_price": prediction,
            "log_return": cache_data.get('log_return'),
            "model_type": f"{model_type.upper()} (Log-Return)",
            "prediction_horizon": horizon_str,
            "change_percentage": change_pct,
            "data_source": data_source,
            "accuracy_metrics": accuracy_metrics if accuracy_metrics else None
        })
    
    except Exception as e:
        logger.error(f"Error getting BERA details: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        })

@app.route('/performance/bera', methods=['GET'])
async def bera_performance():
    """Return performance metrics for BERA model"""
    try:
        token_name = "berausd"
        if token_name not in PREDICTION_PERFORMANCE:
            return jsonify({
                "status": "error",
                "message": "No performance data available for BERA"
            })
        
        # Filter only verified predictions
        verified_predictions = [
            p for p in PREDICTION_PERFORMANCE[token_name] 
            if 'actual_price' in p
        ]
        
        if not verified_predictions:
            return jsonify({
                "status": "success",
                "token": "BERA",
                "message": "No verified predictions yet",
                "pending_predictions": len(PREDICTION_PERFORMANCE[token_name])
            })
        
        # Calculate metrics
        avg_error = sum(p['error'] for p in verified_predictions) / len(verified_predictions)
        avg_error_pct = sum(p['error_pct'] for p in verified_predictions) / len(verified_predictions)
        
        # Direction accuracy (predicted up/down correctly)
        correct_direction = sum(
            1 for p in verified_predictions
            if (p['predicted_price'] > p['latest_price'] and p['actual_price'] > p['latest_price']) or
                (p['predicted_price'] < p['latest_price'] and p['actual_price'] < p['latest_price'])
        )
        direction_accuracy = (correct_direction / len(verified_predictions)) * 100 if verified_predictions else 0
        
        return jsonify({
            "status": "success",
            "token": "BERA",
            "metrics": {
                "verified_predictions": len(verified_predictions),
                "average_error": avg_error,
                "average_error_percent": avg_error_pct,
                "direction_accuracy_percent": direction_accuracy,
                "recent_predictions": verified_predictions[-10:]  # Last 10 predictions
            }
        })
    
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/model_info/bera', methods=['GET'])
async def bera_model_info():
    """Return information about the BERA model"""
    try:
        token_name = "berausd"
        prediction_horizon = TOKEN_INFO[token_name]['prediction_horizon']
        
        # Get comparison data
        comparison_data = load_model_comparison(token_name, prediction_horizon)
        
        # Get current model selection
        best_model = determine_best_model(token_name, prediction_horizon)
        
        return jsonify({
            "status": "success",
            "token": "BERA",
            "model_type": "Log-Return",
            "prediction_horizon": f"{prediction_horizon}m",
            "look_back_window": TOKEN_INFO[token_name]['look_back'],
            "best_model": best_model.upper(),
            "model_path": os.path.join(MODELS_DIR, f'{token_name}_logreturn_{best_model}_model_{prediction_horizon}m.pkl'),
            "comparison_data": comparison_data,
            "model_selection_criteria": "MZTAE (lower is better) & Directional Accuracy (higher is better)"
        })
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        })

##============== Main entry point =============
if __name__ == '__main__':
    # Initialize and display configuration
    logger.info("=" * 50)
    logger.info("BERA PRICE PREDICTION API")
    logger.info("=" * 50)
    
    # Log token info
    logger.info("Model configuration:")
    for token, info in TOKEN_INFO.items():
        logger.info(f"  {info['full_name']} ({token}):")
        logger.info(f"    Model type: {info['model_type']}")
        logger.info(f"    Prediction horizon: {info['prediction_horizon']}m")
        logger.info(f"    Look back window: {info['look_back']} data points")
    
    # Check for model files
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Models directory not found: {MODELS_DIR}")
    else:
        # Count model files
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras') or f.endswith('.pkl')]
        comparison_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.json')]
        
        logger.info(f"Found {len(model_files)} model files and {len(comparison_files)} comparison files")
        
        # Pre-load best model info
        token_name = "berausd"
        prediction_horizon = TOKEN_INFO[token_name]['prediction_horizon']
        best_model = determine_best_model(token_name, prediction_horizon)
        logger.info(f"Best model for BERA ({prediction_horizon}m): {best_model.upper()}")
    
    # Ensure cache directories exist
    os.makedirs(TIINGO_CACHE_DIR, exist_ok=True)
    os.makedirs(OKX_CACHE_DIR, exist_ok=True)
    logger.info(f"Cache directories created: Tiingo={TIINGO_CACHE_DIR}, OKX={OKX_CACHE_DIR}")
    
    # Start the Flask app
    logger.info(f"Starting API server on port {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT)
