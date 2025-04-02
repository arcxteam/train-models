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

def load_model_comparison(json_path):
    with open(json_path, 'r') as f:
        comparison_data = json.load(f)
    return comparison_data

def get_best_model(comparison_data):
    # Pilih model dengan RMSE terendah atau R² tertinggi
    best_model = min(comparison_data, key=lambda x: comparison_data[x]['rmse'])
    return best_model

@app.route('/load_best_model', methods=['GET'])
def load_best_model():
    # Lokasi file JSON perbandingan model
    solusd_json_path = '/root/allora/worker/models/solusd_comparison_5m.json'
    ethusd_json_path = '/root/allora/worker/models/ethusd_comparison_360m.json'
    
    # Memuat perbandingan model dari kedua file JSON
    solusd_comparison_data = load_model_comparison(solusd_json_path)
    ethusd_comparison_data = load_model_comparison(ethusd_json_path)
    
    # Mendapatkan model terbaik untuk SOL dan ETH
    best_model_sol = get_best_model(solusd_comparison_data)
    best_model_eth = get_best_model(ethusd_comparison_data)
    
    # Informasikan model yang dipilih untuk SOL dan ETH
    return jsonify({
        "best_model_sol": best_model_sol,
        "best_model_eth": best_model_eth
    })

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)

# Constants from environment variables
API_PORT = int(os.environ.get('API_PORT', 8000))
LOOK_BACK = int(os.environ.get('LOOK_BACK', 10))  # Default to 10 if not set
PREDICTION_STEPS = int(os.environ.get('PREDICTION_STEPS', 10))  # Default to 10 if not set

# Define prediction horizons for specific tokens
PREDICTION_HORIZONS = {
    'solusd': 5,    # 5-minute prediction for SOL
    'ethusd': 360   # 6-hour prediction for ETH
}

# HTTP Response Codes
HTTP_RESPONSE_CODE_200 = 200
HTTP_RESPONSE_CODE_404 = 404
HTTP_RESPONSE_CODE_500 = 500

# Directory paths
CSV_DIR = os.path.join(DATA_BASE_PATH, 'Binance')

# Cache for best model selections
MODEL_SELECTION_CACHE = {}

# Performance tracking dictionary
PREDICTION_PERFORMANCE = {}

# Function to determine the best model based on comparison JSON files
def determine_best_model(token_name, prediction_horizon):
    """
    Determines the best model for a token based on metrics in comparison JSON files.
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'solusd')
        prediction_horizon (int): Prediction horizon in minutes
        
    Returns:
        str: Model type ('lstm', 'rf', or 'xgb')
    """
    cache_key = f"{token_name}_{prediction_horizon}"
    
    # Return cached result if available
    if cache_key in MODEL_SELECTION_CACHE:
        return MODEL_SELECTION_CACHE[cache_key]
    
    # Look for comparison file
    comparison_file = os.path.join(MODELS_DIR, f"{token_name}_comparison_{prediction_horizon}m.json")
    
    if os.path.exists(comparison_file):
        try:
            with open(comparison_file, 'r') as f:
                comparison_data = json.load(f)
            
            # Find model with lowest RMSE
            if comparison_data:
                best_model = min(comparison_data.items(), key=lambda x: x[1].get('rmse', float('inf')))[0]
                
                logger.info(f"Best model for {token_name} ({prediction_horizon}m) based on comparison: {best_model}")
                logger.info(f"RMSE metrics: {comparison_data[best_model].get('rmse', 'N/A')}")
                
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
    
    # Define file paths based on the selected model type
    if best_model_type == 'lstm':
        model_path = os.path.join(MODELS_DIR, f'{token_name}_model_{prediction_horizon}m.keras')
    else:
        model_path = os.path.join(MODELS_DIR, f'{token_name}_{best_model_type}_model_{prediction_horizon}m.pkl')
    
    scaler_path = os.path.join(MODELS_DIR, f'{token_name}_scaler_{prediction_horizon}m.pkl')
    
    # Attempt to load the best model
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # Load model based on its type
            if best_model_type == 'lstm':
                model = load_model(model_path)
            else:
                model = joblib.load(model_path)
                
            scaler = joblib.load(scaler_path)
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
            model_path = os.path.join(MODELS_DIR, f'{token_name}_model_{prediction_horizon}m.keras')
        else:
            model_path = os.path.join(MODELS_DIR, f'{token_name}_{model_type}_model_{prediction_horizon}m.pkl')
        
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

# Load data from CSV file
def load_data_from_csv(token_name, limit=None):
    csv_path = os.path.join(CSV_DIR, f"{token_name.upper()}.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None, None, None
    
    try:
        df = pd.read_csv(csv_path)
        
        if 'price' not in df.columns:
            logger.error(f"CSV file for {token_name} doesn't have 'price' column")
            return None, None, None
        
        # Sort by timestamp or index to ensure chronological order
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        # Get the most recent data points if limit is specified
        if limit and limit < len(df):
            df = df.tail(limit)
        
        prices = df['price'].values
        block_heights = df['block_height'].values if 'block_height' in df.columns else np.arange(len(prices))
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else np.array([None] * len(prices))
            
        logger.info(f"Loaded {len(prices)} data points from CSV for {token_name}")
        return prices.reshape(-1, 1), block_heights, timestamps
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        logger.error(traceback.format_exc())
        return None, None, None

# Combined data loading function
def load_latest_data(token_name, limit=None):
    # First try database
    prices, block_heights, timestamps = load_data_from_db(token_name, limit)
    
    # If database data not found, try CSV
    if prices is None:
        prices, block_heights, timestamps = load_data_from_csv(token_name, limit)
    
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

# Cache predictions to improve performance
@functools.lru_cache(maxsize=32)
def cached_prediction(token_name, prediction_horizon):
    """Cached prediction function with enhanced logging"""
    start_time = time.time()
    logger.info(f"Starting prediction for {token_name} (horizon: {prediction_horizon}m)")
    
    # Load the selected model and scaler
    model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
    model_load_time = time.time() - start_time
    
    if model is None or scaler is None:
        logger.error(f"Failed to load any model for {token_name}")
        return None
    
    logger.info(f"Loaded {model_type.upper()} model in {model_load_time:.3f}s")
    
    # Load the latest price data
    data_load_start = time.time()
    prices, block_heights, timestamps = load_latest_data(token_name, LOOK_BACK)
    data_load_time = time.time() - data_load_start
    
    if prices is None or len(prices) == 0:
        logger.error(f"No data found for {token_name}")
        return None
    
    # Get the latest
    if len(prices) < LOOK_BACK:
        logger.warning(f"Not enough data points for {token_name}: needed {LOOK_BACK}, got {len(prices)}")
        # Continue with available data if we have at least 5 points
        if len(prices) < 5:
            logger.error("Insufficient data for reliable prediction")
            return None
    
    logger.info(f"Loaded {len(prices)} data points in {data_load_time:.3f}s")
    
    # Get the latest price and block info for logging
    latest_price = float(prices[-1][0])
    latest_block = int(block_heights[-1]) if block_heights is not None and len(block_heights) > 0 else None
    latest_timestamp = timestamps[-1] if timestamps is not None and len(timestamps) > 0 else None
    
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
        
        # Inverse transform to get the actual price
        if model_type == "lstm":
            prediction = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
        else:
            prediction = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
        
        prediction_value = float(prediction[0][0])
        
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
        logger.info(f"  Direction: {direction} ({change_pct:.2f}%)")
        logger.info(f"  Performance: Model Load={model_load_time:.3f}s, Data Load={data_load_time:.3f}s, " +
                    f"Preprocess={preprocess_time:.3f}s, Predict={predict_time:.3f}s, Total={total_prediction_time:.3f}s")
        
        return prediction_value
    except Exception as e:
        logger.error(f"Error during prediction for {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None

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

# Endpoint root ("/")
@app.route('/', methods=['GET'])
def home():
    return "Welcome to Allora Prediction API", 200

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
                if len(parts) >= 2:
                    token = parts[0].upper()
                    
                    # Extract prediction horizon (e.g., 5m, 360m)
                    if 'model' in filename:
                        horizon = parts[2].replace('m.keras', '').replace('m.pkl', '')
                    elif 'scaler' in filename:
                        horizon = parts[2].replace('m.pkl', '')
                    else:
                        continue
                    
                    # Determine model type
                    model_type = 'lstm' if filename.endswith('.keras') else \
                                'xgb' if 'xgb' in filename else \
                                'rf' if 'rf' in filename else 'unknown'
                    
                    if token not in available_models:
                        available_models[token] = {}
                    if horizon not in available_models[token]:
                        available_models[token][horizon] = []
                    if model_type not in available_models[token][horizon]:
                        available_models[token][horizon].append(model_type)
        
        # Add information about best models based on comparison files
        for token in available_models:
            for horizon in available_models[token]:
                token_name = token.lower()
                try:
                    comparison_file = os.path.join(MODELS_DIR, f"{token_name}_comparison_{horizon}m.json")
                    if os.path.exists(comparison_file):
                        with open(comparison_file, 'r') as f:
                            comparison_data = json.load(f)
                        
                        # Find best model
                        if comparison_data:
                            best_model = min(comparison_data.items(), 
                                            key=lambda x: x[1].get('rmse', float('inf')))[0]
                            available_models[token][horizon] = {
                                'models': available_models[token][horizon],
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
        model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
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
            "model_type": model_type.upper(),
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
    
    try:
        comparison_file = os.path.join(MODELS_DIR, f"{token_name}_comparison_{prediction_horizon}m.json")
        if not os.path.exists(comparison_file):
            return jsonify({
                "status": "error",
                "message": f"No comparison data available for {token}"
            })
        
        with open(comparison_file, 'r') as f:
            comparison_data = json.load(f)
        
        # Find best model
        best_model = min(comparison_data.items(), key=lambda x: x[1].get('rmse', float('inf')))[0]
        
        return jsonify({
            "status": "success",
            "token": token.upper(),
            "prediction_horizon": f"{prediction_horizon}m",
            "best_model": best_model,
            "comparison_data": comparison_data
        })
    
    except Exception as e:
        logger.error(f"Error getting comparison data: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error",
            "message": str(e)
        })

if __name__ == '__main__':
    # Initialize and display configuration
    logger.info("=" * 50)
    logger.info("CRYPTO PRICE PREDICTION API - ENHANCED VERSION")
    logger.info("=" * 50)
    
    # Log token horizons
    logger.info("Prediction horizons:")
    for token, horizon in PREDICTION_HORIZONS.items():
        logger.info(f"  {token.upper()}: {horizon} minutes")
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
            best_model = determine_best_model(token, horizon)
            logger.info(f"Best model for {token.upper()} ({horizon}m): {best_model.upper()}")
    
    # Start the Flask app
    logger.info(f"Starting API server on port {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT)