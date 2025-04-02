import numpy as np
import sqlite3
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import logging
import zipfile
import glob
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
from datetime import datetime

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
BINANCE_DIR = os.path.join(DATA_BASE_PATH, 'binance', 'futures-klines')

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

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
        'timestamp': combined_df['timestamp']
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
    # First try to load from DB
    prices, timestamps = load_data_from_db(token_name, hours_back)
    
    # If DB data not found, try loading from Binance files
    if prices is None:
        logger.info(f"Data not found in database for {token_name}, trying Binance files...")
        prices, timestamps = load_data_from_binance(token_name, hours_back)
    
    if prices is None:
        raise ValueError(f"Could not load data for {token_name} from any source")
    
    logger.info(f"Loaded {len(prices)} data points for {token_name}")
    return prices, timestamps

# Calculate directional accuracy
def calculate_directional_accuracy(y_true, y_pred, y_prev):
    """
    Calculate the directional accuracy (how often the model correctly predicts price movement direction)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        y_prev: Previous values (to determine actual direction)
        
    Returns:
        float: Percentage of correctly predicted directions
    """
    # Convert to numpy arrays if needed
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prev = np.array(y_prev)
    
    # Calculate actual and predicted directions
    actual_direction = np.sign(y_true - y_prev)
    predicted_direction = np.sign(y_pred - y_prev)
    
    # Calculate accuracy
    correct_directions = np.sum(actual_direction == predicted_direction)
    total_directions = len(actual_direction)
    
    # Calculate directional accuracy percentage
    directional_accuracy = (correct_directions / total_directions) * 100
    
    return directional_accuracy

# Prepare data for Random Forest and XGBoost with time series consideration
def prepare_data_for_tree_models(data, look_back, prediction_horizon):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, Y, prev_Y = [], [], []
    for i in range(len(scaled_data) - look_back - prediction_horizon):
        X.append(scaled_data[i:(i + look_back), 0])
        Y.append(scaled_data[i + look_back + prediction_horizon - 1, 0])
        prev_Y.append(scaled_data[i + look_back - 1, 0])  # Value right before prediction
    X = np.array(X)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    return X, Y, prev_Y, scaler

# Prepare data for LSTM model with time series consideration
def prepare_data_for_lstm(data, look_back, prediction_horizon):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, Y, prev_Y = [], [], []
    for i in range(len(scaled_data) - look_back - prediction_horizon):
        X.append(scaled_data[i:(i + look_back), 0])
        Y.append(scaled_data[i + look_back + prediction_horizon - 1, 0])
        prev_Y.append(scaled_data[i + look_back - 1, 0])  # Value right before prediction
    X = np.array(X)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    # Reshape input for LSTM [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, Y, prev_Y, scaler

# Train and save Random Forest model
def train_and_save_rf_model(token_name, look_back, prediction_horizon, hours_data=None):
    try:
        logger.info(f"Training Random Forest model for {token_name} with a {prediction_horizon}-minute horizon.")
        
        data, timestamps = load_data(token_name, hours_data)
        X, Y, prev_Y, scaler = prepare_data_for_tree_models(data, look_back, prediction_horizon)
        
        # Use time series split for more realistic evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Use a portion for final testing
        X_train_val, X_test, Y_train_val, Y_test, prev_Y_train_val, prev_Y_test = train_test_split(
            X, Y, prev_Y, test_size=0.3, shuffle=False
        )
        
        # Random Forest model with enhanced hyperparameter tuning
        pipeline = Pipeline([
            ('rf', RandomForestRegressor(random_state=42))
        ])

        # Enhanced hyperparameter grid for better performance
        param_dist = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [None, 10, 20],
            'rf__min_samples_split': [2, 5, 7],
            'rf__min_samples_leaf': [1, 2, 4],
            'rf__max_features': ['sqrt', 'log2', None, 0.4],
            'rf__bootstrap': [True, False],
            'rf__criterion': ['squared_error', 'absolute_error']
        }

        # Use RandomizedSearchCV with TimeSeriesSplit for cross-validation
        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=6, cv=tscv,
            scoring='neg_mean_squared_error', n_jobs=3, random_state=42, verbose=1
        )
        random_search.fit(X_train_val, Y_train_val)

        best_model = random_search.best_estimator_
        logger.info(f"Best RF parameters: {random_search.best_params_}")

        # Making predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model with multiple metrics
        metrics = evaluate_model(Y_test, y_pred, prev_Y_test, "Random Forest")

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
        X, Y, prev_Y, scaler = prepare_data_for_tree_models(data, look_back, prediction_horizon)
        
        # Use time series split for more realistic evaluation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Use a portion for final testing
        X_train_val, X_test, Y_train_val, Y_test, prev_Y_train_val, prev_Y_test = train_test_split(
            X, Y, prev_Y, test_size=0.3, shuffle=False
        )
        
        # XGBoost model with enhanced hyperparameter tuning
        pipeline = Pipeline([
            ('xgb', XGBRegressor(random_state=42, objective='reg:squarederror'))
        ])

        # Enhanced hyperparameter grid for better performance
        param_dist = {
            'xgb__n_estimators': [100, 200, 300],
            'xgb__max_depth': [3, 5, 7],
            'xgb__learning_rate': [0.01, 0.05, 0.1],
            'xgb__subsample': [0.8, 1.0],
            'xgb__colsample_bytree': [0.7, 0.9],
            'xgb__min_child_weight': [2, 5],
            'xgb__gamma': [0, 0.1],
            'xgb__reg_alpha': [0, 0.5],
            'xgb__reg_lambda': [0.5, 1.0]
        }

        # Use RandomizedSearchCV with TimeSeriesSplit for cross-validation
        random_search = RandomizedSearchCV(
            pipeline, param_distributions=param_dist, n_iter=6, cv=tscv,
            scoring='neg_mean_squared_error', n_jobs=3, random_state=42, verbose=1
        )
        random_search.fit(X_train_val, Y_train_val)

        best_model = random_search.best_estimator_
        logger.info(f"Best XGB parameters: {random_search.best_params_}")

        # Making predictions
        y_pred = best_model.predict(X_test)

        # Evaluate the model with multiple metrics
        metrics = evaluate_model(Y_test, y_pred, prev_Y_test, "XGBoost")

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
        X, Y, prev_Y, scaler = prepare_data_for_lstm(data, look_back, prediction_horizon)
        
        # Use a portion for validation during training and a portion for final testing
        X_train, X_temp, Y_train, Y_temp, prev_Y_train, prev_Y_temp = train_test_split(
            X, Y, prev_Y, test_size=0.3, shuffle=False
        )
        X_val, X_test, Y_val, Y_test, prev_Y_val, prev_Y_test = train_test_split(
            X_temp, Y_temp, prev_Y_temp, test_size=0.5, shuffle=False
        )
        
        # Try multiple LSTM architectures
        architectures = [
            {
                'name': 'lstm_simple',
                'units': [50, 25],
                'dropout': 0.2,
                'learning_rate': 0.001,
                'batch_size': 32
            },
            {
                'name': 'lstm_medium',
                'units': [100, 50],
                'dropout': 0.3,
                'learning_rate': 0.001,
                'batch_size': 32
            }
        ]
        
        best_val_loss = float('inf')
        best_model = None
        best_architecture = None
        
        for arch in architectures:
            logger.info(f"Training LSTM architecture: {arch['name']}")
            
            # Create LSTM model based on architecture
            model = Sequential()
            
            # First LSTM layer
            model.add(LSTM(units=arch['units'][0], return_sequences=len(arch['units']) > 1, 
                          input_shape=(look_back, 1)))
            model.add(Dropout(arch['dropout']))
            
            # Middle LSTM layers (if any)
            for i in range(1, len(arch['units']) - 1):
                model.add(LSTM(units=arch['units'][i], return_sequences=True))
                model.add(Dropout(arch['dropout']))
            
            # Last LSTM layer (if more than one)
            if len(arch['units']) > 1:
                model.add(LSTM(units=arch['units'][-1], return_sequences=False))
                model.add(Dropout(arch['dropout']))
            
            # Output layer
            model.add(Dense(units=1))
            
            # Compile the model
            optimizer = Adam(learning_rate=arch['learning_rate'])
            model.compile(optimizer=optimizer, loss='mean_squared_error')
            
            # Callbacks for training
            early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
            
            # Train the model
            history = model.fit(
                X_train, Y_train,
                epochs=50,
                batch_size=arch['batch_size'],
                validation_data=(X_val, Y_val),
                callbacks=[early_stop, reduce_lr],
                verbose=1
            )
            
            # Evaluate on validation set
            val_loss = min(history.history['val_loss'])
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model
                best_architecture = arch
                logger.info(f"New best architecture found: {arch['name']} with val_loss: {val_loss}")
        
        logger.info(f"Best LSTM architecture for {token_name}: {best_architecture['name']}")
        
        # Making predictions with the best model
        y_pred = best_model.predict(X_test)

        # Evaluate the model with multiple metrics
        metrics = evaluate_model(Y_test, y_pred.flatten(), prev_Y_test, "LSTM")
        
        # Add architecture info to metrics
        metrics['architecture'] = best_architecture['name']
        metrics['units'] = best_architecture['units']
        metrics['dropout'] = best_architecture['dropout']
        metrics['learning_rate'] = best_architecture['learning_rate']
        
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
        save_model(best_model, model_path)
        
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

# Enhanced evaluation function with directional accuracy
def evaluate_model(y_true, y_pred, y_prev, model_name):
    """
    Evaluate model with multiple metrics including directional accuracy
    """
    # Standard regression metrics
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    dir_acc = calculate_directional_accuracy(y_true, y_pred, y_prev)
    
    # Calculate percent error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    logger.info(f"{model_name} - Mean Absolute Error: {mae}")
    logger.info(f"{model_name} - Root Mean Squared Error: {rmse}")
    logger.info(f"{model_name} - R^2 Score: {r2}")
    logger.info(f"{model_name} - Directional Accuracy: {dir_acc:.2f}%")
    logger.info(f"{model_name} - Mean Absolute Percentage Error: {mape:.2f}%")
    
    return {
        'model': model_name,
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'directional_accuracy': float(dir_acc),
        'mape': float(mape)
    }

# Compare models and select the best one
def train_and_compare_models(token_name, look_back, prediction_horizon, hours_data=None):
    results = {}
    
    # Train Random Forest
    logger.info(f"Starting RandomForest training for {token_name}")
    rf_metrics = train_and_save_rf_model(token_name, look_back, prediction_horizon, hours_data)
    if rf_metrics:
        results['rf'] = rf_metrics
    
    # Train XGBoost
    logger.info(f"Starting XGBoost training for {token_name}")
    xgb_metrics = train_and_save_xgb_model(token_name, look_back, prediction_horizon, hours_data)
    if xgb_metrics:
        results['xgb'] = xgb_metrics
    
    # Train LSTM
    logger.info(f"Starting LSTM training for {token_name}")
    lstm_metrics = train_and_save_lstm_model(token_name, look_back, prediction_horizon, hours_data)
    if lstm_metrics:
        results['lstm'] = lstm_metrics
    
    # Compare and select the best model
    if results:
        # Create comparison JSON with each model's metrics
        comparison_data = {}
        for model_type, metrics in results.items():
            comparison_data[model_type] = {
                'rmse': metrics['rmse'],
                'mae': metrics['mae'],
                'r2': metrics['r2'],
                'directional_accuracy': metrics['directional_accuracy'],
                'mape': metrics['mape']
            }
        
        # Save comparison results
        comparison_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_comparison_{prediction_horizon}m.json')
        with open(comparison_path, 'w') as f:
            json.dump(comparison_data, f)
        
        # Determine best model based on RMSE and directional accuracy
        # We'll rank models by RMSE (lower is better) but also consider directional accuracy
        best_rmse_model = min(comparison_data.items(), key=lambda x: x[1]['rmse'])[0]
        best_dir_acc_model = max(comparison_data.items(), key=lambda x: x[1]['directional_accuracy'])[0]
        
        # Log the results
        logger.info(f"\nBest model for {token_name} ({prediction_horizon}-minute prediction) by RMSE: {best_rmse_model}")
        logger.info(f"RMSE: {comparison_data[best_rmse_model]['rmse']}, Directional Accuracy: {comparison_data[best_rmse_model]['directional_accuracy']}%")
        
        logger.info(f"Best model for {token_name} ({prediction_horizon}-minute prediction) by Directional Accuracy: {best_dir_acc_model}")
        logger.info(f"RMSE: {comparison_data[best_dir_acc_model]['rmse']}, Directional Accuracy: {comparison_data[best_dir_acc_model]['directional_accuracy']}%")
        
        # Create comparison plots
        plt.figure(figsize=(15, 10))
        
        # RMSE comparison (lower is better)
        plt.subplot(2, 2, 1)
        models = list(comparison_data.keys())
        rmse_values = [comparison_data[model]['rmse'] for model in models]
        plt.bar(models, rmse_values)
        plt.title('RMSE Comparison (Lower is Better)')
        plt.ylabel('RMSE')
        
        # RÂ² comparison (higher is better)
        plt.subplot(2, 2, 2)
        r2_values = [comparison_data[model]['r2'] for model in models]
        plt.bar(models, r2_values)
        plt.title('RÂ² Comparison (Higher is Better)')
        plt.ylabel('RÂ²')
        
        # Directional Accuracy comparison (higher is better)
        plt.subplot(2, 2, 3)
        dir_acc_values = [comparison_data[model]['directional_accuracy'] for model in models]
        plt.bar(models, dir_acc_values)
        plt.title('Directional Accuracy Comparison (Higher is Better)')
        plt.ylabel('Directional Accuracy (%)')
        
        # MAPE comparison (lower is better)
        plt.subplot(2, 2, 4)
        mape_values = [comparison_data[model]['mape'] for model in models]
        plt.bar(models, mape_values)
        plt.title('MAPE Comparison (Lower is Better)')
        plt.ylabel('MAPE (%)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODELS_DIR, f'{token_name.lower()}_comparison_{prediction_horizon}m.png'))
        
        # Return the best model by RMSE as default
        return comparison_data[best_rmse_model]
    
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
    if best_sol_model:
        logger.info(f"SOL/USD (5m): {best_sol_model['model']}")
        logger.info(f"  RMSE: {best_sol_model['rmse']}")
        logger.info(f"  Directional Accuracy: {best_sol_model['directional_accuracy']}%")
    else:
        logger.info("SOL/USD (5m): No model trained")
        
    if best_eth_model:
        logger.info(f"ETH/USD (6h): {best_eth_model['model']}")
        logger.info(f"  RMSE: {best_eth_model['rmse']}")
        logger.info(f"  Directional Accuracy: {best_eth_model['directional_accuracy']}%")
    else:
        logger.info("ETH/USD (6h): No model trained")
    
    logger.info("Training process completed!")