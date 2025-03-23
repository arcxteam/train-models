import numpy as np
import sqlite3
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import app_config for paths
from app_config import DATABASE_PATH, DATA_BASE_PATH

# Base directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CSV_DIR = os.path.join(DATA_BASE_PATH, 'Binance')

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Fetch data from the database
def load_data_from_db(token_name, hours_back=None):
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            if hours_back:
                # Calculate the minimum timestamp based on hours_back
                cursor.execute("""
                    SELECT price, timestamp FROM prices 
                    WHERE token=? AND timestamp >= (SELECT MAX(timestamp) FROM prices WHERE token=?) - ? * 3600
                    ORDER BY block_height ASC
                """, (token_name, token_name, hours_back))
            else:
                cursor.execute("""
                    SELECT price, timestamp FROM prices 
                    WHERE token=?
                    ORDER BY block_height ASC
                """, (token_name,))
                
            result = cursor.fetchall()
            
        # Return both price and timestamp
        if result:
            try:
                prices = np.array([x[0] for x in result]).reshape(-1, 1)
                timestamps = np.array([x[1] for x in result])
                return prices, timestamps
            except Exception as e:
                logger.error(f"Error formatting database results: {e}")
                return None, None
        else:
            logger.warning(f"No data found in database for {token_name}")
            return None, None
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        return None, None

# Fetch data from CSV files in Binance folder
def load_data_from_csv(token_name, hours_back=None):
    csv_path = os.path.join(CSV_DIR, f"{token_name.upper()}.csv")
    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return None, None
    
    try:
        # Load data from CSV
        df = pd.read_csv(csv_path)
        
        # Ensure required columns exist
        required_columns = ['timestamp', 'price']
        if not all(col in df.columns for col in required_columns):
            logger.error(f"CSV file missing required columns. Required: {required_columns}, Found: {df.columns.tolist()}")
            return None, None
        
        # Sort by timestamp to ensure chronological order
        df = df.sort_values('timestamp')
        
        # Filter by hours_back if specified
        if hours_back:
            max_timestamp = df['timestamp'].max()
            min_timestamp = max_timestamp - (hours_back * 3600)
            df = df[df['timestamp'] >= min_timestamp]
        
        prices = df['price'].values.reshape(-1, 1)
        timestamps = df['timestamp'].values
        
        return prices, timestamps
    except Exception as e:
        logger.error(f"Error loading CSV data: {e}")
        return None, None

# Combined data loading function that tries both sources
def load_data(token_name, hours_back=None):
    # First try to load from DB
    prices, timestamps = load_data_from_db(token_name, hours_back)
    
    # If DB data not found, try loading from CSV
    if prices is None:
        logger.info(f"Data not found in database for {token_name}, trying CSV...")
        prices, timestamps = load_data_from_csv(token_name, hours_back)
    
    if prices is None:
        raise ValueError(f"Could not load data for {token_name} from any source")
    
    logger.info(f"Loaded {len(prices)} data points for {token_name}")
    return prices, timestamps

# Prepare data for Random Forest and XGBoost
def prepare_data_for_tree_models(data, look_back, prediction_horizon):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, Y = [], []
    for i in range(len(scaled_data) - look_back - prediction_horizon):
        X.append(scaled_data[i:(i + look_back), 0])
        Y.append(scaled_data[i + look_back + prediction_horizon - 1, 0])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y, scaler

# Prepare data for LSTM model
def prepare_data_for_lstm(data, look_back, prediction_horizon):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, Y = [], []
    for i in range(len(scaled_data) - look_back - prediction_horizon):
        X.append(scaled_data[i:(i + look_back), 0])
        Y.append(scaled_data[i + look_back + prediction_horizon - 1, 0])
    X = np.array(X)
    Y = np.array(Y)
    # Reshape input for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y, scaler

# Train and save Random Forest model
def train_and_save_rf_model(token_name, look_back, prediction_horizon, hours_data=None):
    try:
        logger.info(f"Training Random Forest model for {token_name} with a {prediction_horizon}-minute horizon.")
        
        data, timestamps = load_data(token_name, hours_data)
        X, Y, scaler = prepare_data_for_tree_models(data, look_back, prediction_horizon)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        # Random Forest model with enhanced hyperparameter tuning
        pipeline = Pipeline([
            ('rf', RandomForestRegressor(random_state=42))
        ])

        # Hyperparameter grid - reduced options for faster training
        param_dist = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [None, 10, 20],
            'rf__min_samples_split': [2, 5]
        }

        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=7, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=3, random_state=42, verbose=1
        )
        random_search.fit(X_train, Y_train)

        best_model = random_search.best_estimator_
        logger.info(f"Best RF parameters: {random_search.best_params_}")

        # Making predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        metrics = evaluate_model(Y_test, y_pred, "Random Forest")

        # Save the model
        model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_rf_model_{prediction_horizon}m.pkl')
        joblib.dump(best_model, model_path)
        
        # Save the scaler
        scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_scaler_{prediction_horizon}m.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Save metrics
        metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_rf_metrics_{prediction_horizon}m.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

        logger.info(f"RF Model for {token_name} ({prediction_horizon}-minute prediction) saved to {model_path}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error occurred while training Random Forest model for {token_name}: {e}", exc_info=True)
        return None

# Train and save XGBoost model
def train_and_save_xgb_model(token_name, look_back, prediction_horizon, hours_data=None):
    try:
        logger.info(f"Training XGBoost model for {token_name} with a {prediction_horizon}-minute horizon.")
        
        data, timestamps = load_data(token_name, hours_data)
        X, Y, scaler = prepare_data_for_tree_models(data, look_back, prediction_horizon)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        # XGBoost model with hyperparameter tuning
        pipeline = Pipeline([
            ('xgb', XGBRegressor(random_state=42, objective='reg:squarederror'))
        ])

        # Hyperparameter grid - reduced options for faster training
        param_dist = {
            'xgb__n_estimators': [100, 200],
            'xgb__max_depth': [3, 6],
            'xgb__learning_rate': [0.01, 0.1],
            'xgb__subsample': [0.8, 1.0]
        }

        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=7, cv=3, 
            scoring='neg_mean_squared_error', n_jobs=3, random_state=42, verbose=1
        )
        random_search.fit(X_train, Y_train)

        best_model = random_search.best_estimator_
        logger.info(f"Best XGB parameters: {random_search.best_params_}")

        # Making predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model
        metrics = evaluate_model(Y_test, y_pred, "XGBoost")

        # Save the model
        model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_xgb_model_{prediction_horizon}m.pkl')
        joblib.dump(best_model, model_path)
        
        # Save the scaler
        scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_scaler_{prediction_horizon}m.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Save metrics
        metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_xgb_metrics_{prediction_horizon}m.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

        logger.info(f"XGB Model for {token_name} ({prediction_horizon}-minute prediction) saved to {model_path}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error occurred while training XGBoost model for {token_name}: {e}", exc_info=True)
        return None

# Train and save LSTM model
def train_and_save_lstm_model(token_name, look_back, prediction_horizon, hours_data=None):
    try:
        logger.info(f"Training LSTM model for {token_name} with a {prediction_horizon}-minute horizon.")
        
        data, timestamps = load_data(token_name, hours_data)
        X, Y, scaler = prepare_data_for_lstm(data, look_back, prediction_horizon)
        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        
        # Create LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train the model
        history = model.fit(
            X_train, Y_train,
            epochs=70,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stop],
            verbose=1
        )
        
        # Making predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        metrics = evaluate_model(Y_test, y_pred.flatten(), "LSTM")
        
        # Plot training & validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title(f'LSTM Model Loss for {token_name} ({prediction_horizon}m)')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.savefig(os.path.join(MODELS_DIR, f'{token_name.lower()}_lstm_loss_{prediction_horizon}m.png'))
        
        # Save the model in .keras format
        model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_model_{prediction_horizon}m.keras')
        save_model(model, model_path)
        
        # Save the scaler
        scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_scaler_{prediction_horizon}m.pkl')
        joblib.dump(scaler, scaler_path)
        
        # Save metrics
        metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_lstm_metrics_{prediction_horizon}m.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)

        logger.info(f"LSTM Model for {token_name} ({prediction_horizon}-minute prediction) saved to {model_path}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error occurred while training LSTM model for {token_name}: {e}", exc_info=True)
        return None

# Evaluate model function
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    logger.info(f"{model_name} - Mean Absolute Error: {mae}")
    logger.info(f"{model_name} - Mean Squared Error: {mse}")
    logger.info(f"{model_name} - Root Mean Squared Error: {rmse}")
    logger.info(f"{model_name} - R^2 Score: {r2}")
    
    return {
        'model': model_name,
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2)
    }

# Compare models and select the best one
def train_and_compare_models(token_name, look_back, prediction_horizon, hours_data=None):
    results = []
    
    # Train Random Forest
    logger.info(f"Starting RandomForest training for {token_name}")
    rf_metrics = train_and_save_rf_model(token_name, look_back, prediction_horizon, hours_data)
    if rf_metrics:
        results.append(rf_metrics)
    
    # Train XGBoost
    logger.info(f"Starting XGBoost training for {token_name}")
    xgb_metrics = train_and_save_xgb_model(token_name, look_back, prediction_horizon, hours_data)
    if xgb_metrics:
        results.append(xgb_metrics)
    
    # Train LSTM
    logger.info(f"Starting LSTM training for {token_name}")
    lstm_metrics = train_and_save_lstm_model(token_name, look_back, prediction_horizon, hours_data)
    if lstm_metrics:
        results.append(lstm_metrics)
    
    # Compare and select the best model
    if results:
        # Sort by RMSE (lower is better)
        results.sort(key=lambda x: x['rmse'])
        best_model = results[0]
        
        logger.info(f"\nBest model for {token_name} ({prediction_horizon}-minute prediction): {best_model['model']}")
        logger.info(f"RMSE: {best_model['rmse']}, R²: {best_model['r2']}")
        
        # Save comparison results
        comparison_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_comparison_{prediction_horizon}m.json')
        with open(comparison_path, 'w') as f:
            json.dump(results, f)
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        
        # RMSE comparison
        plt.subplot(1, 2, 1)
        models = [result['model'] for result in results]
        rmse_values = [result['rmse'] for result in results]
        plt.bar(models, rmse_values)
        plt.title('RMSE Comparison (Lower is Better)')
        plt.ylabel('RMSE')
        
        # R² comparison
        plt.subplot(1, 2, 2)
        r2_values = [result['r2'] for result in results]
        plt.bar(models, r2_values)
        plt.title('R² Comparison (Higher is Better)')
        plt.ylabel('R²')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f'{token_name.lower()}_comparison_{prediction_horizon}m.png'))
        
        return best_model
    
    logger.warning(f"No successful models trained for {token_name}")
    return None

# Define different time horizons for model training
time_horizons = {
    '5m': (15, 5),      # 5-minute prediction with 15 data points lookback
    '360m': (48, 360),  # 6-hour (360 min) prediction with 48 data points lookback
}

# Define how much historical data to use (in hours)
data_requirements = {
    '5m': 168,      # 7 days * 24 hours
    '360m': 2160,   # 90 days * 24 hours
}

# Main execution
if __name__ == "__main__":
    logger.info("Starting model training process")
    
    # Check if necessary directories exist
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created models directory at {MODELS_DIR}")
    
    # Train models for SOL/USD with 5-minute horizon
    token_name = "solusd"
    horizon_name = "5m"
    look_back, prediction_horizon = time_horizons[horizon_name]
    hours_data = data_requirements[horizon_name]
    logger.info(f"\n{'='*50}")
    logger.info(f"Training models for {token_name.upper()} with {prediction_horizon}-minute horizon")
    logger.info(f"{'='*50}")
    best_sol_model = train_and_compare_models(token_name, look_back, prediction_horizon, hours_data)
    
    # Train models for ETH/USD with 6-hour horizon
    token_name = "ethusd"
    horizon_name = "360m"
    look_back, prediction_horizon = time_horizons[horizon_name]
    hours_data = data_requirements[horizon_name]
    logger.info(f"\n{'='*50}")
    logger.info(f"Training models for {token_name.upper()} with {prediction_horizon}-minute horizon")
    logger.info(f"{'='*50}")
    best_eth_model = train_and_compare_models(token_name, look_back, prediction_horizon, hours_data)
    
    # Print summary of best models
    logger.info("\nSummary of Best Models:")
    logger.info(f"SOL/USD (5m): {best_sol_model['model'] if best_sol_model else 'No model trained'}")
    logger.info(f"ETH/USD (6h): {best_eth_model['model'] if best_eth_model else 'No model trained'}")
    
    logger.info("Training process completed!")