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
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Import app_config for paths
from app_config import DATABASE_PATH, DATA_BASE_PATH

app = Flask(__name__)

# Get base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Constants from environment variables
API_PORT = int(os.environ.get('API_PORT', 8000))
LOOK_BACK = int(os.environ.get('LOOK_BACK', 60))  # Default to 60 for log-return model
PREDICTION_STEPS = int(os.environ.get('PREDICTION_STEPS', 60))  # Default to 60 for 1-hour prediction

# Define prediction horizons for specific tokens
PREDICTION_HORIZONS = {
    'berausd': 60,     # 1-hour prediction for BERA log-return
    'ethusd': 360      # 6-hour prediction for ETH volatility
}

# Define model types for specific tokens
MODEL_TYPES = {
    'berausd': 'logreturn',   # BERA uses log-return model
    'ethusd': 'volatility'    # ETH uses volatility model
}

# HTTP Response Codes
HTTP_RESPONSE_CODE_200 = 200
HTTP_RESPONSE_CODE_404 = 404
HTTP_RESPONSE_CODE_500 = 500

# Directory paths
CSV_DIR = os.path.join(DATA_BASE_PATH, 'binance', 'futures-klines')

# Cache for best model selections
MODEL_SELECTION_CACHE = {}

# Performance tracking dictionary
PREDICTION_PERFORMANCE = {}

# Cache untuk prediksi log-return dan volatilitas
LOG_RETURN_CACHE = {}
VOLATILITY_CACHE = {}

def load_model_comparison(json_path):
    """Load model comparison data from JSON file"""
    try:
        with open(json_path, 'r') as f:
            comparison_data = json.load(f)
        return comparison_data
    except Exception as e:
        logger.error(f"Error loading comparison file {json_path}: {e}")
        return None

def get_best_model(comparison_data, model_type='logreturn'):
    """Get the best model based on appropriate metric for the model type"""
    if not comparison_data:
        return 'rf'  # Default to Random Forest if no comparison data
    
    if model_type == 'logreturn':
        # For log-return models, use MZTAE (lower is better)
        best_model = min(comparison_data.items(), key=lambda x: x[1].get('mztae', float('inf')))[0]
    else:
        # For volatility models, use RMSE (lower is better)
        best_model = min(comparison_data.items(), key=lambda x: x[1].get('rmse', float('inf')))[0]
    
    return best_model

# Function to determine the best model based on comparison JSON files
def determine_best_model(token_name, prediction_horizon):
    """
    Determines the best model for a token based on metrics in comparison JSON files.
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        prediction_horizon (int): Prediction horizon in minutes
        
    Returns:
        str: Model type ('lstm', 'rf', or 'xgb')
    """
    cache_key = f"{token_name}_{prediction_horizon}"
    
    # Return cached result if available
    if cache_key in MODEL_SELECTION_CACHE:
        return MODEL_SELECTION_CACHE[cache_key]
    
    # Get model type (logreturn or volatility)
    model_type = MODEL_TYPES.get(token_name, 'logreturn')
    
    # Look for comparison file based on model type
    comparison_file = os.path.join(MODELS_DIR, f"{token_name}_{model_type}_comparison_{prediction_horizon}m.json")
    
    if os.path.exists(comparison_file):
        try:
            comparison_data = load_model_comparison(comparison_file)
            if comparison_data:
                best_model = get_best_model(comparison_data, model_type)
                
                if model_type == 'logreturn':
                    logger.info(f"Best model for {token_name} ({prediction_horizon}m) based on comparison: {best_model}")
                    logger.info(f"MZTAE: {comparison_data[best_model].get('mztae', 'N/A')}, Dir Acc: {comparison_data[best_model].get('directional_accuracy', 'N/A')}%")
                else:
                    logger.info(f"Best model for {token_name} ({prediction_horizon}m) based on comparison: {best_model}")
                    logger.info(f"RMSE: {comparison_data[best_model].get('rmse', 'N/A')}, MAPE: {comparison_data[best_model].get('mape', 'N/A')}%")
                
                # Cache the result
                MODEL_SELECTION_CACHE[cache_key] = best_model
                return best_model
        except Exception as e:
            logger.error(f"Error loading comparison file for {token_name}: {e}")
    
    # If no comparison file or parsing error, default to Random Forest
    logger.warning(f"No comparison file found for {token_name} ({prediction_horizon}m), defaulting to 'rf'")
    MODEL_SELECTION_CACHE[cache_key] = 'rf'
    return 'rf'

# Load model and scaler for a specific token
def load_model_and_scaler(token_name, prediction_horizon):
    # Determine the best model to load first
    best_model_type = determine_best_model(token_name, prediction_horizon)
    
    # Get whether this is a logreturn or volatility model
    model_category = MODEL_TYPES.get(token_name, 'logreturn')
    
    # Define file paths based on the selected model type
    if best_model_type == 'lstm':
        model_path = os.path.join(MODELS_DIR, f'{token_name}_{model_category}_model_{prediction_horizon}m.keras')
    else:
        model_path = os.path.join(MODELS_DIR, f'{token_name}_{model_category}_{best_model_type}_model_{prediction_horizon}m.pkl')
    
    scaler_path = os.path.join(MODELS_DIR, f'{token_name}_{model_category}_scaler_{prediction_horizon}m.pkl')
    
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
            logger.info(f"Successfully loaded {best_model_type.upper()} {model_category} model for {token_name}")
            return model, scaler, best_model_type, model_category
        except Exception as e:
            logger.error(f"Error loading {best_model_type} model for {token_name}: {e}")
    else:
        logger.warning(f"Best model files not found for {token_name} ({best_model_type})")
    
    # If best model fails, try other models in fallback order
    fallback_order = ['rf', 'xgb', 'lstm'] if best_model_type != 'rf' else ['xgb', 'lstm']
    
    for model_type in fallback_order:
        if model_type == 'lstm':
            model_path = os.path.join(MODELS_DIR, f'{token_name}_{model_category}_model_{prediction_horizon}m.keras')
        else:
            model_path = os.path.join(MODELS_DIR, f'{token_name}_{model_category}_{model_type}_model_{prediction_horizon}m.pkl')
        
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                if model_type == 'lstm':
                    model = load_model(model_path)
                else:
                    model = joblib.load(model_path)
                    
                scaler = joblib.load(scaler_path)
                logger.info(f"Loaded fallback {model_type.upper()} {model_category} model for {token_name}")
                return model, scaler, model_type, model_category
            except Exception as e:
                logger.error(f"Error loading fallback {model_type} model for {token_name}: {e}")
    
    logger.error(f"No working models found for {token_name} with {prediction_horizon}m horizon")
    return None, None, None, None

# Load data from SQLite database
def load_data_from_db(token_name, limit=None):
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if timestamp column exists in the schema
            cursor.execute("PRAGMA table_info(prices)")
            columns = cursor.fetchall()
            has_timestamp = any(col[1] == 'timestamp' for col in columns)
            
            if has_timestamp:
                query = """
                    SELECT price, block_height, timestamp FROM prices 
                    WHERE token=?
                    ORDER BY block_height DESC
                """
            else:
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
            
            if has_timestamp and data.shape[1] > 2:
                timestamps = data[:, 2]
            else:
                timestamps = np.array([None] * len(prices))
                
            logger.info(f"Loaded {len(prices)} data points from database for {token_name}")
            return prices, block_heights, timestamps
        else:
            logger.warning(f"No data found in database for {token_name}")
            return None, None, None
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

# Function to load data directly from Binance CSV files
def load_data_from_binance(token_name, limit=None):
    """
    Load the latest price data from Binance CSV files
    """
    # Convert token to proper Binance format
    binance_token = token_name.replace('usd', 'usdt').lower()
    token_dir = os.path.join(CSV_DIR, binance_token)
    
    if not os.path.exists(token_dir):
        logger.warning(f"Binance data directory not found: {token_dir}")
        return None, None, None
    
    try:
        # Find all CSV files
        csv_files = [f for f in os.listdir(token_dir) if f.endswith('.csv')]
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

# Combined data loading function
def load_latest_data(token_name, limit=None):
    """
    Load the latest data from either database or Binance files
    """
    # First try database
    prices, block_heights, timestamps = load_data_from_db(token_name, limit)
    
    # If database data not found, try Binance CSV files
    if prices is None:
        prices, block_heights, timestamps = load_data_from_binance(token_name, limit)
    
    return prices, block_heights, timestamps

# Track prediction performance for analysis
def track_prediction(token_name, predicted_price, latest_price, block_height):
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

# Check prediction against actual price once data becomes available
def verify_prediction(token_name, block_height, predicted_price, actual_price):
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

# Get the latest block height from the database
def get_latest_block_height():
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(block_height) FROM prices")
            result = cursor.fetchone()
        
        if result and result[0]:
            return result[0]
        else:
            logger.warning("No block height found in database")
            return None
    except Exception as e:
        logger.error(f"Error getting latest block height: {e}")
        return None

# Get price at specific block height
def get_price_at_block(token_name, block_height):
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

# Calculate log returns for prediction
def calculate_log_returns(prices, period=60):
    """
    Calculate log returns for the given prices with specified period.
    """
    # Ensure prices is a flattened array
    if prices.ndim > 1:
        prices = prices.flatten()
    
    # Calculate log returns: ln(price_t+period / price_t)
    log_returns = np.log(prices[period:] / prices[:-period])
    
    return log_returns

# Calculate volatility for prediction
def calculate_volatility(prices, window=360):
    """
    Calculate volatility as standardized rolling standard deviation of log returns.
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
    
    # Standardize
    standardized_volatility = volatility * np.sqrt(360)
    
    return standardized_volatility

# Prepare data for prediction based on model type
def prepare_prediction_data(prices, look_back, model_category='logreturn', prediction_horizon=60):
    """
    Prepare the latest data for prediction based on model category
    """
    if model_category == 'logreturn':
        # For log-return models, we just need the last look_back prices
        if len(prices) < look_back:
            logger.warning(f"Not enough price data for log-return prediction: needed {look_back}, got {len(prices)}")
            return None
        
        # Get the last look_back prices
        prediction_data = prices[-look_back:].flatten()
        
        return prediction_data.reshape(1, -1)
    
    elif model_category == 'volatility':
        # For volatility models, we need at least window + look_back data points
        volatility_window = 360  # Standard window for volatility calculation
        
        if len(prices) < look_back + volatility_window:
            logger.warning(f"Not enough price data for volatility prediction: needed {look_back + volatility_window}, got {len(prices)}")
            return None
        
        # Get the last look_back prices
        prediction_data = prices[-look_back:].flatten()
        
        return prediction_data.reshape(1, -1)
    
    else:
        logger.error(f"Unknown model category: {model_category}")
        return None

# Cache predictions to improve performance
@functools.lru_cache(maxsize=32)
def cached_prediction(token_name, prediction_horizon):
    """Cached prediction function with enhanced logging"""
    start_time = time.time()
    logger.info(f"Starting prediction for {token_name} (horizon: {prediction_horizon}m)")
    
    # Define the look_back parameter based on the model type
    look_back = 60  # Default for BERA log-return model
    if token_name == 'ethusd':
        look_back = 72  # ETH volatility model uses 72 data points lookback
    
    # Load the selected model and scaler
    model, scaler, model_type, model_category = load_model_and_scaler(token_name, prediction_horizon)
    model_load_time = time.time() - start_time
    
    if model is None or scaler is None:
        logger.error(f"Failed to load any model for {token_name}")
        return None
    
    logger.info(f"Loaded {model_type.upper()} {model_category} model in {model_load_time:.3f}s")
    
    # Load the latest price data - make sure we get enough data points
    data_load_start = time.time()
    
    # For log-return models, we just need look_back data points
    # For volatility models, we need look_back + volatility window data points
    data_points_needed = look_back
    if model_category == 'volatility':
        data_points_needed += 360  # Add volatility window
    
    prices, block_heights, timestamps = load_latest_data(token_name, data_points_needed)
    data_load_time = time.time() - data_load_start
    
    if prices is None or len(prices) == 0:
        logger.error(f"No data found for {token_name}")
        return None
    
    # Get the latest price and block info for logging
    latest_price = float(prices[-1][0])
    latest_block = int(block_heights[-1]) if block_heights is not None and len(block_heights) > 0 else None
    latest_timestamp = timestamps[-1] if timestamps is not None and len(timestamps) > 0 else None
    
    logger.info(f"Loaded {len(prices)} data points in {data_load_time:.3f}s")
    
    # Preprocess data based on model type
    preprocess_start = time.time()
    
    prediction_data = prepare_prediction_data(prices, look_back, model_category, prediction_horizon)
    if prediction_data is None:
        logger.error(f"Failed to prepare prediction data for {token_name}")
        return None
    
    # Scale the data
    scaled_data = scaler.transform(prediction_data.reshape(1, -1))
    preprocess_time = time.time() - preprocess_start
    
    # Make prediction based on model type
    try:
        predict_start = time.time()
        
        if model_type == "lstm":
            # Reshape for LSTM [samples, time steps, features]
            X_pred = scaled_data.reshape(1, look_back, 1)
            pred = model.predict(X_pred, verbose=0)
        else:  # Tree-based models (RF, XGB)
            # Use as is for tree models
            pred = model.predict(scaled_data)
        
        predict_time = time.time() - predict_start
        
        # Process prediction based on model category
        if model_category == 'logreturn':
            # For log-return models, pred is the predicted log return
            log_return = pred[0] if isinstance(pred, np.ndarray) else pred
            
            # Convert log return to actual price: price_t+horizon = price_t * exp(log_return)
            prediction_value = latest_price * np.exp(log_return)
            
            # Ensure it's a scalar
            if hasattr(prediction_value, '__len__'):
                prediction_value = prediction_value[0]
        
        elif model_category == 'volatility':
            # For volatility models, pred is the predicted volatility
            volatility = pred[0] if isinstance(pred, np.ndarray) else pred
            
            # For reporting purposes, convert volatility to price movement as percentage
            # This is just a rough approximation for display
            vol_pct = volatility * 100  # Convert to percentage
            
            # Simulate a price based on volatility (this is just for display)
            # In reality, volatility prediction doesn't tell direction
            # Just simulate a price range based on 1 standard deviation
            prediction_value = latest_price * (1 + vol_pct/100)
            
            # Log the volatility prediction
            logger.info(f"Predicted volatility: {vol_pct:.2f}%")
        
        # Calculate prediction direction and percentage
        direction = "UP" if prediction_value > latest_price else "DOWN"
        change_pct = ((prediction_value - latest_price) / latest_price) * 100
        
        # Track the prediction for later verification
        track_prediction(token_name, prediction_value, latest_price, latest_block)
        
        # Calculate total prediction time
        total_prediction_time = time.time() - start_time
        
        # Log comprehensive prediction info
        logger.info(f"=== Prediction for {token_name.upper()} (Model: {model_type.upper()} {model_category}) ===")
        logger.info(f"  Horizon: {prediction_horizon} minutes")
        logger.info(f"  Block: {latest_block}, Timestamp: {latest_timestamp}")
        logger.info(f"  Latest Price: ${latest_price:.4f}")
        logger.info(f"  Predicted Price: ${prediction_value:.4f}")
        logger.info(f"  Direction: {direction} ({change_pct:.2f}%)")
        logger.info(f"  Performance: Model Load={model_load_time:.3f}s, Data Load={data_load_time:.3f}s, " +
                    f"Preprocess={preprocess_time:.3f}s, Predict={predict_time:.3f}s, Total={total_prediction_time:.3f}s")
        
        return prediction_value
    except Exception as e:
        logger.error(f"Error during prediction for {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None

# Fungsi untuk cache prediksi log-return
@functools.lru_cache(maxsize=32)
def cached_log_return_prediction(token_name, prediction_horizon):
    """Cached log-return prediction function"""
    start_time = time.time()
    logger.info(f"Starting log-return prediction for {token_name} (horizon: {prediction_horizon}m)")
    
    # Load the selected model and scaler
    model, scaler, model_type, model_category = load_model_and_scaler(token_name, prediction_horizon)
    model_load_time = time.time() - start_time
    
    if model is None or scaler is None or model_category != 'logreturn':
        logger.error(f"Failed to load log-return model for {token_name}")
        return None
    
    logger.info(f"Loaded {model_type.upper()} {model_category} model in {model_load_time:.3f}s")
    
    # Define look_back parameter for log-return model
    look_back = 60  # Default for log-return model
    
    # Load the latest price data
    data_load_start = time.time()
    prices, block_heights, timestamps = load_latest_data(token_name, look_back)
    data_load_time = time.time() - data_load_start
    
    if prices is None or len(prices) < look_back:
        logger.error(f"Insufficient data for {token_name} log-return prediction: needed {look_back}, got {len(prices) if prices is not None else 0}")
        return None
    
    # Get the latest price for reference
    latest_price = float(prices[-1][0])
    
    # Preprocess data
    preprocess_start = time.time()
    scaled_data = scaler.transform(prices)
    preprocess_time = time.time() - preprocess_start
    
    # Make prediction based on model type
    try:
        predict_start = time.time()
        
        if model_type == "lstm":
            # Reshape for LSTM [samples, time steps, features]
            X_pred = scaled_data.reshape(1, scaled_data.shape[0], 1)
            pred = model.predict(X_pred, verbose=0)
        else:  # Tree-based models (RF, XGB)
            # Reshape for tree models [samples, features]
            X_pred = scaled_data.reshape(1, -1)
            pred = model.predict(X_pred)
        
        predict_time = time.time() - predict_start
        
        # For log-return models, pred is the log return value
        log_return = pred[0] if isinstance(pred, np.ndarray) else pred
        
        # Calculate the predicted price
        predicted_price = latest_price * np.exp(log_return)
        
        # Calculate total prediction time
        total_prediction_time = time.time() - start_time
        
        # Store in cache
        LOG_RETURN_CACHE[token_name] = {
            'log_return': float(log_return),
            'predicted_price': float(predicted_price) if hasattr(predicted_price, '__len__') else predicted_price,
            'latest_price': latest_price,
            'timestamp': time.time(),
            'model_type': model_type
        }
        
        # Log results
        logger.info(f"Log-Return prediction for {token_name}: {log_return:.6f}")
        logger.info(f"Predicted price: ${predicted_price:.4f} (from ${latest_price:.4f})")
        logger.info(f"Performance: Total={total_prediction_time:.3f}s")
        
        return float(log_return)
    
    except Exception as e:
        logger.error(f"Error during log-return prediction for {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None

# Fungsi untuk cache prediksi volatilitas
@functools.lru_cache(maxsize=32)
def cached_volatility_prediction(token_name, prediction_horizon):
    """Cached volatility prediction function"""
    start_time = time.time()
    logger.info(f"Starting volatility prediction for {token_name} (horizon: {prediction_horizon}m)")
    
    # Load the selected model and scaler
    model, scaler, model_type, model_category = load_model_and_scaler(token_name, prediction_horizon)
    model_load_time = time.time() - start_time
    
    if model is None or scaler is None or model_category != 'volatility':
        logger.error(f"Failed to load volatility model for {token_name}")
        return None
    
    logger.info(f"Loaded {model_type.upper()} {model_category} model in {model_load_time:.3f}s")
    
    # Define parameters for volatility model
    look_back = 72  # For volatility model
    volatility_window = 360  # Standard window for volatility calculation
    
    # Load the latest price data - make sure we get enough data points
    data_load_start = time.time()
    prices, block_heights, timestamps = load_latest_data(token_name, look_back + volatility_window)
    data_load_time = time.time() - data_load_start
    
    if prices is None or len(prices) < look_back + volatility_window:
        logger.error(f"Insufficient data for {token_name} volatility prediction: needed {look_back + volatility_window}, got {len(prices) if prices is not None else 0}")
        return None
    
    # Get the latest price for reference
    latest_price = float(prices[-1][0])
    
    # Preprocess data
    preprocess_start = time.time()
    volatility_data = prices[-(look_back + volatility_window):]
    prediction_data = volatility_data[-look_back:].reshape(1, -1)
    scaled_data = scaler.transform(prediction_data)
    preprocess_time = time.time() - preprocess_start
    
    # Make prediction based on model type
    try:
        predict_start = time.time()
        
        if model_type == "lstm":
            # Reshape for LSTM [samples, time steps, features]
            X_pred = scaled_data.reshape(1, look_back, 1)
            pred = model.predict(X_pred, verbose=0)
        else:  # Tree-based models (RF, XGB)
            # Use as is for tree models
            pred = model.predict(scaled_data)
        
        predict_time = time.time() - predict_start
        
        # For volatility models, pred is the volatility value
        volatility = pred[0] if isinstance(pred, np.ndarray) else pred
        
        # Convert volatility to percentage for display
        volatility_pct = float(volatility * 100)
        
        # Calculate total prediction time
        total_prediction_time = time.time() - start_time
        
        # Store in cache
        VOLATILITY_CACHE[token_name] = {
            'volatility': float(volatility),
            'volatility_pct': volatility_pct,
            'price_range_low': latest_price * (1 - volatility),
            'price_range_high': latest_price * (1 + volatility),
            'latest_price': latest_price,
            'timestamp': time.time(),
            'model_type': model_type
        }
        
        # Log results
        logger.info(f"Volatility prediction for {token_name}: {volatility_pct:.2f}%")
        logger.info(f"Price range: ${latest_price * (1 - volatility):.4f} - ${latest_price * (1 + volatility):.4f}")
        logger.info(f"Performance: Total={total_prediction_time:.3f}s")
        
        return float(volatility)
    
    except Exception as e:
        logger.error(f"Error during volatility prediction for {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None

# Endpoint root ("/")
@app.route('/', methods=['GET'])
def home():
    return "Welcome to Cryptocurrency Prediction API", 200

# Endpoint health check ("/health")
@app.route('/health', methods=['GET'])
async def health_check():
    """Simple health check endpoint"""
    return "OK", 200

@app.route('/inference/<token>', methods=['GET'])
async def get_inference(token):
    """
    Get price prediction for a token
    """
    logger.info(f"Received inference request for {token}")
    if not token:
        response = json.dumps({"error": "Token is required"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = f"{token}USD".lower()
    
    # Get the appropriate prediction horizon for this token
    prediction_horizon = PREDICTION_HORIZONS.get(token_name, PREDICTION_STEPS)
    
    try:
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, cached_prediction, token_name, prediction_horizon)

        if prediction is None:
            response = json.dumps({"error": "No data found or model unavailable for the specified token"})
            return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')

        logger.info(f"{token} inference result: {prediction}")
        return Response(str(prediction), status=HTTP_RESPONSE_CODE_200, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        logger.error(traceback.format_exc())
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/predict_price/<token>', methods=['GET'])
async def predict_price(token):
    """
    API endpoint for Allora Network compatibility
    Returns a simple price prediction
    """
    logger.info(f"Received price prediction request for {token}")
    if not token:
        response = json.dumps({"error": "Token is required"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = f"{token}USD".lower()
    
    # Get the appropriate prediction horizon for this token
    prediction_horizon = PREDICTION_HORIZONS.get(token_name, PREDICTION_STEPS)
    
    try:
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, cached_prediction, token_name, prediction_horizon)

        if prediction is None:
            response = json.dumps({"error": "No data found or model unavailable for the specified token"})
            return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')

        logger.info(f"{token} prediction result: {prediction}")
        return Response(str(prediction), status=HTTP_RESPONSE_CODE_200, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error processing price prediction: {e}")
        logger.error(traceback.format_exc())
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/log_return/<token>', methods=['GET'])
async def get_log_return(token):
    """
    Get log-return prediction for a token
    """
    logger.info(f"Received log-return request for {token}")
    if not token:
        response = json.dumps({"error": "Token is required"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = f"{token}USD".lower()
    
    # Check if token supports log-return model
    model_type = MODEL_TYPES.get(token_name)
    if model_type != 'logreturn':
        response = json.dumps({"error": f"Log-return model not available for {token}"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    # Get the appropriate prediction horizon
    prediction_horizon = PREDICTION_HORIZONS.get(token_name, 60)  # Default to 60m for log-return
    
    try:
        loop = asyncio.get_event_loop()
        log_return = await loop.run_in_executor(None, cached_log_return_prediction, token_name, prediction_horizon)

        if log_return is None:
            response = json.dumps({"error": "Failed to calculate log-return prediction"})
            return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

        # Get additional data from cache
        cache_data = LOG_RETURN_CACHE.get(token_name, {})
        
        # Return as JSON
        result = {
            "token": token.upper(),
            "log_return": log_return,
            "predicted_price": cache_data.get('predicted_price'),
            "latest_price": cache_data.get('latest_price'),
            "prediction_horizon": f"{prediction_horizon}m",
            "model_type": cache_data.get('model_type', 'unknown').upper(),
            "timestamp": datetime.fromtimestamp(cache_data.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing log-return request: {e}")
        logger.error(traceback.format_exc())
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/volatility/<token>', methods=['GET'])
async def get_volatility(token):
    """
    Get volatility prediction for a token
    """
    logger.info(f"Received volatility request for {token}")
    if not token:
        response = json.dumps({"error": "Token is required"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = f"{token}USD".lower()
    
    # Check if token supports volatility model
    model_type = MODEL_TYPES.get(token_name)
    if model_type != 'volatility':
        response = json.dumps({"error": f"Volatility model not available for {token}"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    # Get the appropriate prediction horizon
    prediction_horizon = PREDICTION_HORIZONS.get(token_name, 360)  # Default to 360m for volatility
    
    try:
        loop = asyncio.get_event_loop()
        volatility = await loop.run_in_executor(None, cached_volatility_prediction, token_name, prediction_horizon)

        if volatility is None:
            response = json.dumps({"error": "Failed to calculate volatility prediction"})
            return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

        # Get additional data from cache
        cache_data = VOLATILITY_CACHE.get(token_name, {})
        
        # Return as JSON
        result = {
            "token": token.upper(),
            "volatility": volatility,
            "volatility_percentage": cache_data.get('volatility_pct'),
            "price_range": {
                "low": cache_data.get('price_range_low'),
                "current": cache_data.get('latest_price'),
                "high": cache_data.get('price_range_high')
            },
            "prediction_horizon": f"{prediction_horizon}m",
            "model_type": cache_data.get('model_type', 'unknown').upper(),
            "timestamp": datetime.fromtimestamp(cache_data.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error processing volatility request: {e}")
        logger.error(traceback.format_exc())
        response = json.dumps({"error": str(e)})
        return Response(response, status=HTTP_RESPONSE_CODE_500, mimetype='application/json')

@app.route('/truth/<token>/<block_height>', methods=['GET'])
async def get_truth(token, block_height):
    """
    Endpoint for ground truth data at specific block height
    Used by Allora Network reputers
    """
    logger.info(f"Received truth request for {token} at block {block_height}")
    
    try:
        block_height = int(block_height)
    except ValueError:
        response = json.dumps({"error": "Invalid block height format"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    token_name = token.lower()
    
    price = get_price_at_block(token_name, block_height)
    
    if price is None:
        response = json.dumps({"error": "No data found for the specified token and block height"})
        return Response(response, status=HTTP_RESPONSE_CODE_404, mimetype='application/json')
    
    logger.info(f"{token} truth at block {block_height}: {price}")
    return Response(str(price), status=HTTP_RESPONSE_CODE_200, mimetype='text/plain')

@app.route('/models', methods=['GET'])
async def list_models():
    """List all available models with their metrics if available"""
    available_models = {}
    
    if not os.path.exists(MODELS_DIR):
        return jsonify({"status": "error", "message": "Models directory not found"})
    
    try:
        # First get all model files
        for filename in os.listdir(MODELS_DIR):
            if filename.endswith('.pkl') or filename.endswith('.keras'):
                parts = filename.split('_')
                if len(parts) >= 3:
                    token = parts[0].upper()
                    model_type = parts[1]  # logreturn or volatility
                    
                    # Extract prediction horizon (e.g., 5m, 360m)
                    if 'model' in filename:
                        horizon = parts[-1].replace('m.keras', '').replace('m.pkl', '')
                    elif 'scaler' in filename:
                        horizon = parts[-1].replace('m.pkl', '')
                    else:
                        continue
                    
                    # Determine algorithm type
                    algo_type = 'lstm' if filename.endswith('.keras') else \
                               'xgb' if 'xgb' in filename else \
                               'rf' if 'rf' in filename else 'unknown'
                    
                    if token not in available_models:
                        available_models[token] = {}
                    if horizon not in available_models[token]:
                        available_models[token][horizon] = {}
                    if model_type not in available_models[token][horizon]:
                        available_models[token][horizon][model_type] = []
                    if algo_type not in available_models[token][horizon][model_type]:
                        available_models[token][horizon][model_type].append(algo_type)
        
        # Add information about best models based on comparison files
        for token in available_models:
            for horizon in available_models[token]:
                for model_type in available_models[token][horizon]:
                    token_name = token.lower()
                    try:
                        comparison_file = os.path.join(MODELS_DIR, f"{token_name}_{model_type}_comparison_{horizon}m.json")
                        if os.path.exists(comparison_file):
                            with open(comparison_file, 'r') as f:
                                comparison_data = json.load(f)
                            
                            # Find best model based on model type
                            if comparison_data:
                                if model_type == 'logreturn':
                                    best_model = min(comparison_data.items(), 
                                                    key=lambda x: x[1].get('mztae', float('inf')))[0]
                                else:  # volatility
                                    best_model = min(comparison_data.items(), 
                                                    key=lambda x: x[1].get('rmse', float('inf')))[0]
                                
                                available_models[token][horizon][model_type] = {
                                    'models': available_models[token][horizon][model_type],
                                    'best_model': best_model,
                                    'metrics': comparison_data
                                }
                    except Exception as e:
                        logger.error(f"Error reading comparison file for {token_name}: {e}")
        
        return jsonify({
            "status": "success",
            "available_models": available_models
        })
    
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/details/<token>', methods=['GET'])
async def token_details(token):
    """Get detailed prediction for a token"""
    if not token:
        return jsonify({"status": "error", "message": "Token is required"})
    
    token_name = f"{token}USD".lower()
    prediction_horizon = PREDICTION_HORIZONS.get(token_name, PREDICTION_STEPS)
    
    try:
        # Get the current price and latest data
        prices, block_heights, timestamps = load_latest_data(token_name, 1)
        if prices is None or len(prices) == 0:
            return jsonify({"status": "error", "message": "No current price data available"})
            
        current_price = float(prices[0][0])
        current_block = int(block_heights[0]) if block_heights is not None and len(block_heights) > 0 else None
        
        # Get the model type
        model, scaler, algo_type, model_category = load_model_and_scaler(token_name, prediction_horizon)
        if model is None:
            return jsonify({"status": "error", "message": f"No model found for {token}"})
        
        # Get prediction
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, cached_prediction, token_name, prediction_horizon)
        
        if prediction is None:
            return jsonify({"status": "error", "message": "Failed to make prediction"})
        
        # Calculate change percentage
        change_pct = ((prediction - current_price) / current_price * 100)
        
        # Format prediction horizon
        horizon_str = f"{prediction_horizon}m"
        if prediction_horizon == 360:
            horizon_str = "6h"
            
        # Get prediction accuracy metrics if available
        accuracy_metrics = {}
        for records in PREDICTION_PERFORMANCE.values():
            verified_records = [r for r in records if 'actual_price' in r]
            if verified_records:
                avg_error = sum(r['error'] for r in verified_records) / len(verified_records)
                avg_error_pct = sum(r['error_pct'] for r in verified_records) / len(verified_records)
                accuracy_metrics = {
                    'average_error': avg_error,
                    'average_error_percent': avg_error_pct,
                    'verified_predictions': len(verified_records)
                }
        
        return jsonify({
            "status": "success",
            "token": token.upper(),
            "current_price": current_price,
            "current_block": current_block,
            "predicted_price": prediction,
            "model_type": f"{algo_type.upper()} ({model_category})",
            "prediction_horizon": horizon_str,
            "change_percentage": change_pct,
            "accuracy_metrics": accuracy_metrics if accuracy_metrics else None
        })
    
    except Exception as e:
        logger.error(f"Error getting token details: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        })

@app.route('/performance', methods=['GET'])
async def model_performance():
    """Return performance metrics for all models"""
    try:
        # Get token from query param, if provided
        token = request.args.get('token')
        
        if token:
            token_name = f"{token}USD".lower()
            if token_name not in PREDICTION_PERFORMANCE:
                return jsonify({
                    "status": "error",
                    "message": f"No performance data available for {token}"
                })
            
            # Filter only verified predictions
            verified_predictions = [
                p for p in PREDICTION_PERFORMANCE[token_name] 
                if 'actual_price' in p
            ]
            
            if not verified_predictions:
                return jsonify({
                    "status": "success",
                    "token": token.upper(),
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
                "token": token.upper(),
                "metrics": {
                    "verified_predictions": len(verified_predictions),
                    "average_error": avg_error,
                    "average_error_percent": avg_error_pct,
                    "direction_accuracy_percent": direction_accuracy,
                    "recent_predictions": verified_predictions[-10:]  # Last 10 predictions
                }
            })
        else:
            # Return metrics for all tokens
            all_metrics = {}
            for token_name, predictions in PREDICTION_PERFORMANCE.items():
                verified_predictions = [p for p in predictions if 'actual_price' in p]
                if verified_predictions:
                    avg_error = sum(p['error'] for p in verified_predictions) / len(verified_predictions)
                    avg_error_pct = sum(p['error_pct'] for p in verified_predictions) / len(verified_predictions)
                    all_metrics[token_name] = {
                        "verified_predictions": len(verified_predictions),
                        "average_error": avg_error,
                        "average_error_percent": avg_error_pct
                    }
            
            return jsonify({
                "status": "success",
                "metrics": all_metrics
            })
    
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/comparison/<token>', methods=['GET'])
async def model_comparison(token):
    """Return comparison data between different models for a token"""
    if not token:
        return jsonify({"status": "error", "message": "Token is required"})
    
    token_name = f"{token}USD".lower()
    prediction_horizon = PREDICTION_HORIZONS.get(token_name, PREDICTION_STEPS)
    model_type = MODEL_TYPES.get(token_name, 'logreturn')
    
    try:
        comparison_file = os.path.join(MODELS_DIR, f"{token_name}_{model_type}_comparison_{prediction_horizon}m.json")
        if not os.path.exists(comparison_file):
            return jsonify({
                "status": "error",
                "message": f"No comparison data available for {token}"
            })
        
        with open(comparison_file, 'r') as f:
            comparison_data = json.load(f)
        
        # Find best model based on model type
        if model_type == 'logreturn':
            best_model = min(comparison_data.items(), key=lambda x: x[1].get('mztae', float('inf')))[0]
            key_metric = 'mztae'
            key_metric_name = 'MZTAE (Modified Z-transformed Absolute Error)'
        else:  # volatility
            best_model = min(comparison_data.items(), key=lambda x: x[1].get('rmse', float('inf')))[0]
            key_metric = 'rmse'
            key_metric_name = 'RMSE (Root Mean Square Error)'
        
        return jsonify({
            "status": "success",
            "token": token.upper(),
            "model_type": model_type,
            "prediction_horizon": f"{prediction_horizon}m",
            "best_model": best_model,
            "key_metric": key_metric_name,
            "comparison_data": comparison_data
        })
    
    except Exception as e:
        logger.error(f"Error getting comparison data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/load_best_model', methods=['GET'])
def load_best_model():
    """
    Endpoint to load and display the best models for BERA and ETH.
    Useful for initialization and configuration checks.
    """
    try:
        # Get the relevant comparison files
        bera_json_path = os.path.join(MODELS_DIR, 'berausd_logreturn_comparison_60m.json')
        eth_json_path = os.path.join(MODELS_DIR, 'ethusd_volatility_comparison_360m.json')
        
        # Load comparison data
        bera_comparison_data = None
        eth_comparison_data = None
        
        if os.path.exists(bera_json_path):
            bera_comparison_data = load_model_comparison(bera_json_path)
        
        if os.path.exists(eth_json_path):
            eth_comparison_data = load_model_comparison(eth_json_path)
        
        # Get best models
        best_model_bera = get_best_model(bera_comparison_data, 'logreturn') if bera_comparison_data else 'N/A'
        best_model_eth = get_best_model(eth_comparison_data, 'volatility') if eth_comparison_data else 'N/A'
        
        # Get key metrics
        bera_metrics = {}
        eth_metrics = {}
        
        if bera_comparison_data and best_model_bera in bera_comparison_data:
            bera_metrics = {
                'mztae': bera_comparison_data[best_model_bera].get('mztae', 'N/A'),
                'directional_accuracy': bera_comparison_data[best_model_bera].get('directional_accuracy', 'N/A')
            }
        
        if eth_comparison_data and best_model_eth in eth_comparison_data:
            eth_metrics = {
                'rmse': eth_comparison_data[best_model_eth].get('rmse', 'N/A'),
                'mape': eth_comparison_data[best_model_eth].get('mape', 'N/A')
            }
        
        # Return the results
        return jsonify({
            "status": "success",
            "bera": {
                "best_model": best_model_bera,
                "model_type": "logreturn",
                "prediction_horizon": "60m",
                "metrics": bera_metrics
            },
            "eth": {
                "best_model": best_model_eth,
                "model_type": "volatility",
                "prediction_horizon": "360m",
                "metrics": eth_metrics
            }
        })
    except Exception as e:
        logger.error(f"Error loading best models: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        })

## =================== Main Function ================= ##

if __name__ == '__main__':
    # Initialize and display configuration
    logger.info("=" * 50)
    logger.info("CRYPTO PRICE PREDICTION API - COMBINED VERSION")
    logger.info("=" * 50)
    
    # Log token horizons
    logger.info("Prediction horizons:")
    for token, horizon in PREDICTION_HORIZONS.items():
        logger.info(f"  {token.upper()}: {horizon} minutes ({MODEL_TYPES.get(token, 'logreturn')} model)")
    logger.info(f"Default prediction horizon: {PREDICTION_STEPS} minutes")
    
    # Check for model files
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Models directory not found: {MODELS_DIR}")
    else:
        # Count model files
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.keras') or f.endswith('.pkl')]
        comparison_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.json')]
        
        logger.info(f"Found {len(model_files)} model files and {len(comparison_files)} comparison files")
        
        # Pre-load best model info for each token
        for token, horizon in PREDICTION_HORIZONS.items():
            model_type = MODEL_TYPES.get(token, 'logreturn')
            best_model = determine_best_model(token, horizon)
            logger.info(f"Best model for {token.upper()} ({horizon}m, {model_type}): {best_model.upper()}")
    
    # Start the Flask app
    logger.info(f"Starting API server on port {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT)