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
from flask import Flask, Response, jsonify, request
from tensorflow.keras.models import load_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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

# Load model and scaler for a specific token
def load_model_and_scaler(token_name, prediction_horizon):
    # First try to load LSTM model (preferred)
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_model_{prediction_horizon}m.keras')
    scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_scaler_{prediction_horizon}m.pkl')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = load_model(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded LSTM model for {token_name}")
            return model, scaler, "lstm"
        except Exception as e:
            logger.error(f"Error loading LSTM model for {token_name}: {e}")
    
    # If LSTM fails or doesn't exist, try XGBoost
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_xgb_model_{prediction_horizon}m.pkl')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded XGBoost model for {token_name}")
            return model, scaler, "xgb"
        except Exception as e:
            logger.error(f"Error loading XGBoost model for {token_name}: {e}")
    
    # If XGBoost fails or doesn't exist, try Random Forest
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_rf_model_{prediction_horizon}m.pkl')
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"Loaded Random Forest model for {token_name}")
            return model, scaler, "rf"
        except Exception as e:
            logger.error(f"Error loading Random Forest model for {token_name}: {e}")
    
    logger.error(f"No models found for {token_name} with {prediction_horizon}m horizon")
    return None, None, None

# Load data from SQLite database
def load_data_from_db(token_name, limit=None):
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            if limit:
                cursor.execute("""
                    SELECT price FROM prices 
                    WHERE token=?
                    ORDER BY block_height DESC
                    LIMIT ?
                """, (token_name, limit))
            else:
                cursor.execute("""
                    SELECT price FROM prices 
                    WHERE token=?
                    ORDER BY block_height DESC
                """, (token_name,))
                
            result = cursor.fetchall()
            
        if result:
            # Convert to numpy array and reverse to get chronological order
            prices = np.array([x[0] for x in result])
            prices = prices[::-1]  # Reverse to get chronological order
            return prices.reshape(-1, 1)
        else:
            logger.warning(f"No data found in database for {token_name}")
            return None
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        return None

# Load data from CSV file
def load_data_from_csv(token_name, limit=None):
    csv_path = os.path.join(CSV_DIR, f"{token_name.upper()}.csv")
    
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        
        if 'price' not in df.columns:
            logger.error(f"CSV file for {token_name} doesn't have 'price' column")
            return None
        
        # Sort by timestamp or index to ensure chronological order
        if 'timestamp' in df.columns:
            df = df.sort_values('timestamp')
        
        prices = df['price'].values
        
        if limit and limit < len(prices):
            prices = prices[-limit:]  # Get the most recent prices
            
        return prices.reshape(-1, 1)
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return None

# Combined data loading function
def load_latest_data(token_name, limit=None):
    # First try database
    prices = load_data_from_db(token_name, limit)
    
    # If database data not found, try CSV
    if prices is None:
        prices = load_data_from_csv(token_name, limit)
    
    return prices

# Cache predictions to improve performance
@functools.lru_cache(maxsize=32)
def cached_prediction(token_name, prediction_horizon):
    model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
    
    if model is None or scaler is None:
        logger.error(f"Failed to load model or scaler for {token_name}")
        return None
    
    # Load the latest price data
    prices = load_latest_data(token_name, LOOK_BACK)
    
    if prices is None or len(prices) == 0:
        logger.error(f"No data found for {token_name}")
        return None
    
    if len(prices) < LOOK_BACK:
        logger.error(f"Not enough data points for {token_name}: needed {LOOK_BACK}, got {len(prices)}")
        return None
    
    # Preprocess data
    scaled_data = scaler.transform(prices)
    
    # Make prediction based on model type
    try:
        if model_type == "lstm":
            # Reshape for LSTM [samples, time steps, features]
            X_pred = scaled_data.reshape(1, scaled_data.shape[0], 1)
            pred = model.predict(X_pred, verbose=0)
        else:  # Tree-based models (RF, XGB)
            # Reshape for tree models [samples, features]
            X_pred = scaled_data.reshape(1, -1)
            pred = model.predict(X_pred)
        
        # Inverse transform to get the actual price
        if model_type == "lstm":
            prediction = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
        else:
            prediction = scaler.inverse_transform(np.array(pred).reshape(-1, 1))
        
        return float(prediction[0][0])
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
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
            return float(result[0])
        else:
            logger.warning(f"No price found for {token_name} at block {block_height}")
            return None
    except Exception as e:
        logger.error(f"Error getting price at block: {e}")
        return None

@app.route('/', methods=['GET'])
async def health():
    return "Crypto Price Prediction API - Status: Healthy"

@app.route('/inference/<token>', methods=['GET'])
async def get_inference(token):
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
    """List all available models"""
    available_models = {}
    
    if not os.path.exists(MODELS_DIR):
        return jsonify({"status": "error", "message": "Models directory not found"})
    
    try:
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
        # Get the model type
        model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
        if model is None:
            return jsonify({"status": "error", "message": f"No model found for {token}"})
        
        # Get the current price
        prices = load_latest_data(token_name, 1)
        current_price = float(prices[0][0]) if prices is not None and len(prices) > 0 else None
        
        # Get prediction
        loop = asyncio.get_event_loop()
        prediction = await loop.run_in_executor(None, cached_prediction, token_name, prediction_horizon)
        
        if prediction is None:
            return jsonify({"status": "error", "message": "Failed to make prediction"})
        
        # Calculate change percentage
        change_pct = ((prediction - current_price) / current_price * 100) if current_price else None
        
        # Format prediction horizon
        horizon_str = f"{prediction_horizon}m"
        if prediction_horizon == 360:
            horizon_str = "6h"
        
        return jsonify({
            "status": "success",
            "token": token.upper(),
            "current_price": current_price,
            "predicted_price": prediction,
            "model_type": model_type.upper(),
            "prediction_horizon": horizon_str,
            "change_percentage": change_pct
        })
    
    except Exception as e:
        logger.error(f"Error getting token details: {e}")
        logger.error(traceback.format_exc())
        return jsonify({
            "status": "error", 
            "message": str(e)
        })

if __name__ == '__main__':
    # Ensure the prediction horizons are set correctly
    logger.info(f"Starting API server with the following prediction horizons:")
    for token, horizon in PREDICTION_HORIZONS.items():
        logger.info(f"  {token.upper()}: {horizon} minutes")
    logger.info(f"Default prediction step: {PREDICTION_STEPS} minutes")
    
    app.run(host='0.0.0.0', port=API_PORT)