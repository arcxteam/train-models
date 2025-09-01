import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import json
import time
import traceback
import sys
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from skopt import BayesSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from scipy.stats import binomtest, pearsonr
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="skopt.space.space")

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Import config untuk path
try:
    from config import DATA_BASE_PATH, TIINGO_DATA_DIR
except ImportError:
    logger.warning("Gagal mengimpor dari config. Mengatur path default.")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_BASE_PATH = os.path.join(BASE_DIR, 'data')
    TIINGO_DATA_DIR = os.path.join(DATA_BASE_PATH, 'tiingo_data')

# Direktori dasar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Pastikan direktori models dan cache ada
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TIINGO_DATA_DIR, exist_ok=True)

def load_tiingo_data(token_name, cache_dir=TIINGO_DATA_DIR):
    """
    Memuat data OHLCV dari file cache Tiingo JSON per token
    Args:
        token_name (str): Nama token (misalnya 'btcusd')
        cache_dir (str): Direktori cache Tiingo
    Returns:
        pd.DataFrame: DataFrame dengan data OHLCV atau None jika gagal
    """
    base_ticker = token_name.lower()
    cache_path = os.path.join(cache_dir, f"tiingo_data_5min_{base_ticker}.json")
    
    if not os.path.exists(cache_path):
        logger.error(f"Cache file not found: {cache_path}")
        return None
        
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data['priceData'])
        
        # Validasi kolom asli Tiingo
        required_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'volumeNotional', 'tradesDone']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} missing in cache, filling with 0.0")
                df[col] = 0.0
        
        # Konversi date ke Timestamp
        df['date'] = pd.to_datetime(df['date'])
        df['timestamp'] = df['date'].astype(np.int64) // 10**9
        df = df.drop_duplicates(subset=['date']).sort_values('date').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} records for {token_name} from Tiingo cache, date range: {df['date'].min()} to {df['date'].max()}")
        return df
        
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError loading Tiingo cache data: {e}")
        logger.error(f"Cache file content may be corrupted: {cache_path}")
        try:
            with open(cache_path, 'r') as f:
                logger.error(f"Cache content sample: {f.read()[:100]}")
        except Exception as read_err:
            logger.error(f"Failed to read cache content: {read_err}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"Error loading Tiingo cache data: {e}")
        logger.error(traceback.format_exc())
        return None

def get_data_snapshot(token_name):
    """
    Ambil snapshot data terkini untuk pelatihan
    - Memuat data dari cache
    - Membersihkan data (handle NaN, inf, dll)
    - Mengembalikan DataFrame yang siap digunakan
    """
    logger.info(f"Mengambil snapshot data untuk {token_name}")
    df = load_tiingo_data(token_name)

    if df is None:
        logger.error(f"Gagal memuat snapshot data untuk {token_name}")
        return None

    # Bersihkan data snapshot
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    if df.isna().any().any():
        logger.error(f"Data snapshot mengandung NaN setelah pembersihan: {df.columns[df.isna().any()]}")
        return None

    logger.info(f"Snapshot data berhasil diambil: {len(df)} records")
    return df

def calculate_log_returns(prices, period=480):
    """
    Menghitung log returns dengan alignment yang benar
    Return: array dengan length yang sama seperti prices, NaN untuk period pertama
    """
    if isinstance(prices, np.ndarray):
        if prices.ndim > 1:
            prices = prices.flatten()
    elif isinstance(prices, pd.Series):
        prices = prices.values
    else:
        prices = np.array(prices)
    
    if len(prices) <= period:
        logger.error(f"Tidak cukup titik data ({len(prices)}) untuk periode ({period})")
        return np.array([])
    
    # Hitung log returns yang benar
    log_returns = np.log(prices[period:] / prices[:-period])
    
    # Kembalikan array dengan length yang sama seperti prices, dengan NaN untuk period pertama
    result = np.full(len(prices), np.nan)
    result[period:] = log_returns
    
    return result

def calculate_zptae(y_true, y_pred, sigma=None, alpha=0.5, window_size=100):
    """
    Menghitung Z-transformed Power-Tanh Absolute Error (ZPTAE) dengan optimasi.
    Args:
        y_true (numpy.array): Nilai aktual
        y_pred (numpy.array): Nilai prediksi
        sigma (float, optional): Standar deviasi untuk normalisasi
        alpha (float): Parameter power-law untuk PowerTanh (default 0.5)
        window_size (int): Ukuran jendela untuk perhitungan std (default 100)
    Returns:
        float: Skor ZPTAE (lebih rendah lebih baik)
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Pastikan length sama
    if len(y_true) != len(y_pred):
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]
        logger.warning(f"y_true and y_pred length mismatch, using first {min_len} elements")
    
    abs_errors = np.abs(y_true - y_pred)
    abs_errors = np.clip(abs_errors, 0, np.percentile(abs_errors, 99))  # Batasi outlier
    n = len(abs_errors)
    zptae_values = np.zeros(n)
    
    # Jika sigma diberikan, gunakan untuk seluruh data
    if sigma is not None:
        sigma_i = sigma
        for i in range(n):
            z_error = abs_errors[i] / sigma_i
            zptae_values[i] = np.sign(z_error) * np.tanh(np.power(np.abs(z_error), alpha))
        return np.mean(zptae_values)
    
    # Hitung standar deviasi bergerak
    for i in range(n):
        start = max(0, i - window_size + 1)
        window = y_true[start:i+1]  # Ambil jendela data aktual
        if len(window) > 1:
            sigma_i = np.std(window)
        else:
            sigma_i = 1e-8  # Nilai kecil untuk menghindari pembagian nol
        
        if sigma_i < 1e-8:
            sigma_i = 1e-8
            
        z_error = abs_errors[i] / sigma_i
        zptae_values[i] = np.sign(z_error) * np.tanh(np.power(np.abs(z_error), alpha))
    
    return np.mean(zptae_values)

def smooth_predictions(preds, alpha=0.1):
    """
    Exponential Moving Average (EMA) smoothing untuk prediksi
    Args:
        preds (numpy.array): Prediksi untuk dihaluskan
        alpha (float): Faktor smoothing (0-1)
    Returns:
        numpy.array: Prediksi yang dihaluskan
    """
    smoothed = [preds[0]]
    for p in preds[1:]:
        smoothed.append(alpha * p + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def prepare_data_for_log_return(data_df, look_back=60, prediction_horizon=480):
    """
    Menyiapkan data untuk pelatihan model prediksi log-return
    PERBAIKAN: Alignment yang benar antara features dan target
    """
    logger.info(f"Menyiapkan data dengan look_back={look_back}, prediction_horizon={prediction_horizon}")
    
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Input harus berupa pandas DataFrame")
    
    required_cols = ['close', 'open', 'high', 'low', 'volume', 'volumeNotional']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Kolom yang diperlukan tidak ada: {missing_cols}")
    
    df = data_df.copy()
    
    # Validasi jumlah data
    min_required = prediction_horizon + look_back + 100
    if len(df) < min_required:
        logger.error(f"Data tidak cukup! Hanya {len(df)} records, butuh minimal {min_required}")
        return None, None, None, None
    
    # Hitung log returns dengan alignment benar
    price_array = df['close'].values
    all_log_returns = calculate_log_returns(price_array, period=prediction_horizon)
    
    # Tambahkan log returns ke DataFrame untuk memudahkan alignment
    df['log_return_target'] = all_log_returns
    
    # PERBAIKAN: Hindari data leakage - split data sebelum feature engineering
    test_size = int(0.2 * len(df))
    if test_size == 0:
        logger.error("Data tidak cukup untuk pembagian train/test")
        return None, None, None, None
        
    train_df = df.iloc[:-test_size].copy()
    test_df = df.iloc[-test_size:].copy()
    
    # Feature engineering hanya pada training data untuk menghindari leakage
    def create_features(data_frame):
        df_temp = data_frame.copy()
        
        # Technical indicators dengan handle NaN
        # MACD
        ema8 = df_temp['close'].ewm(span=8, adjust=False, min_periods=1).mean()
        ema21 = df_temp['close'].ewm(span=21, adjust=False, min_periods=1).mean()
        df_temp['macd_line'] = ema8 - ema21
        df_temp['macd_signal'] = df_temp['macd_line'].ewm(span=14, adjust=False, min_periods=1).mean()
        df_temp['macd_histogram'] = df_temp['macd_line'] - df_temp['macd_signal']
        
        # RSI
        delta = df_temp['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df_temp['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma20 = df_temp['close'].rolling(window=20, min_periods=1).mean()
        std20 = df_temp['close'].rolling(window=20, min_periods=1).std()
        df_temp['bb_upper'] = sma20 + (std20 * 2)
        df_temp['bb_lower'] = sma20 - (std20 * 2)
        df_temp['bb_percent'] = (df_temp['close'] - df_temp['bb_lower']) / (df_temp['bb_upper'] - df_temp['bb_lower'] + 1e-10)
        
        # Volume indicators
        df_temp['volume_ma'] = df_temp['volume'].rolling(window=20, min_periods=1).mean()
        df_temp['volume_ratio'] = df_temp['volume'] / (df_temp['volume_ma'] + 1e-10)
        
        # Price momentum
        df_temp['returns_1'] = df_temp['close'].pct_change(1)
        df_temp['returns_5'] = df_temp['close'].pct_change(5)
        df_temp['returns_10'] = df_temp['close'].pct_change(10)
        
        # Volatility
        df_temp['volatility_20'] = df_temp['close'].rolling(window=20, min_periods=1).std()
        
        # Handle NaN values
        df_temp = df_temp.ffill().bfill()
        
        # Fill remaining NaN dengan nilai netral
        df_temp['rsi'] = df_temp['rsi'].fillna(50)
        df_temp['bb_percent'] = df_temp['bb_percent'].fillna(0.5)
        df_temp['macd_line'] = df_temp['macd_line'].fillna(0)
        df_temp['macd_signal'] = df_temp['macd_signal'].fillna(0)
        df_temp['macd_histogram'] = df_temp['macd_histogram'].fillna(0)
        
        # Remove temporary columns
        if 'volume_ma' in df_temp.columns:
            df_temp = df_temp.drop(['volume_ma'], axis=1)
        
        return df_temp
    
    # Apply feature engineering
    train_df = create_features(train_df)
    test_df = create_features(test_df)
    
    # Feature columns
    feature_columns = [
        'close', 'open', 'high', 'low', 'volume',
        'macd_line', 'macd_signal', 'macd_histogram',
        'rsi', 'bb_percent', 'volume_ratio',
        'returns_1', 'returns_5', 'returns_10',
        'volatility_20'
    ]
    
    # Handle missing values akhir
    train_df = train_df.ffill().bfill()
    test_df = test_df.ffill().bfill()
    
    # Prepare sequences dengan alignment YANG BENAR
    X_train, Y_train, prev_Y_train = [], [], []
    X_test, Y_test, prev_Y_test = [], [], []
    
    # PERBAIKAN: Training data dengan alignment benar
    # Untuk setiap titik i, kita menggunakan features [i-look_back:i] untuk memprediksi target pada i
    for i in range(look_back, len(train_df)):
        if not np.isnan(train_df['log_return_target'].iloc[i]):
            # Features: data dari i-look_back sampai i-1
            window_df = train_df.iloc[i - look_back:i]
            window_features = window_df[feature_columns].values.flatten()
            
            # Target: log return pada waktu i
            X_train.append(window_features)
            Y_train.append(train_df['log_return_target'].iloc[i])
            prev_Y_train.append(window_df['close'].iloc[-1])
    
    # PERBAIKAN: Test data dengan alignment sama
    for i in range(look_back, len(test_df)):
        if not np.isnan(test_df['log_return_target'].iloc[i]):
            window_df = test_df.iloc[i - look_back:i]
            window_features = window_df[feature_columns].values.flatten()
            
            X_test.append(window_features)
            Y_test.append(test_df['log_return_target'].iloc[i])
            prev_Y_test.append(window_df['close'].iloc[-1])
    
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    prev_Y_train = np.array(prev_Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    prev_Y_test = np.array(prev_Y_test)
    
    # VALIDATION CHECKS - sangat penting!
    logger.info(f"Training data: X={X_train.shape}, Y={Y_train.shape}")
    logger.info(f"Test data: X={X_test.shape}, Y={Y_test.shape}")
    
    # Check untuk NaN values
    if len(X_train) == 0 or len(X_test) == 0:
        logger.error("Data training atau testing kosong setelah preparation")
        return None, None, None, None
    
    logger.info(f"Y_train NaN count: {np.sum(np.isnan(Y_train))}")
    logger.info(f"Y_test NaN count: {np.sum(np.isnan(Y_test))}")
    logger.info(f"Y_train stats: mean={np.nanmean(Y_train):.6f}, std={np.nanstd(Y_train):.6f}")
    logger.info(f"Y_test stats: mean={np.nanmean(Y_test):.6f}, std={np.nanstd(Y_test):.6f}")
    
    # Pastikan tidak semua target NaN
    if np.all(np.isnan(Y_train)) or len(Y_train) == 0:
        logger.error("SEMUA TARGET TRAINING NaN ATAU KOSONG!")
        return None, None, None, None
        
    if np.all(np.isnan(Y_test)) or len(Y_test) == 0:
        logger.error("SEMUA TARGET TESTING NaN ATAU KOSONG!")
        return None, None, None, None
    
    return (X_train, Y_train, prev_Y_train, 
            X_test, Y_test, prev_Y_test, 
            feature_columns)

def train_and_save_xgb_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon):
    """
    Melatih dan menyimpan model XGBoost untuk prediksi log-return
    
    Args:
        token_name (str): Nama token (misalnya 'btcusd')
        X_train (numpy.array): Fitur pelatihan
        Y_train (numpy.array): Target pelatihan (log returns)
        X_test (numpy.array): Fitur pengujian
        Y_test (numpy.array): Target pengujian (log returns)
        prev_Y_test (numpy.array): Harga sebelumnya untuk perhitungan akurasi arah
        scaler (MinMaxScaler): Scaler yang telah di-fit
        prediction_horizon (int): Horizon prediksi dalam menit
        
    Returns:
        dict: Metrik untuk model yang dilatih
    """
    logger.info(f"Melatih model XGBoost untuk {token_name}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    pipeline = Pipeline([('xgb', XGBRegressor(random_state=42, objective='reg:squarederror', verbosity=0))])

    param_space = {
        'xgb__n_estimators': [50, 150],
        'xgb__max_depth': [5, 8, 13],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__subsample': [0.5, 1.0],
        'xgb__colsample_bytree': [0.5, 1.0],
        'xgb__min_child_weight': [1, 7],
        'xgb__max_delta_step': [0, 5],
        'xgb__gamma': [0, 1],
        'xgb__reg_lambda': [0, 5],
        'xgb__reg_alpha': [0, 5],
        'xgb__booster': ['gbtree', 'gblinear']
    }
    
    search = BayesSearchCV(
        pipeline, param_space, n_iter=7, cv=tscv,
        scoring='neg_mean_absolute_error', n_jobs=3, random_state=42, verbose=0
    )

    # Pelatihan model
    logger.info("Starting BayesSearchCV fitting for XGBRegressor...")
    search.fit(X_train, Y_train)
    logger.info("BayesSearchCV fitting completed for XGBRegressor.")
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    zptae_score = calculate_zptae(Y_test, y_pred, alpha=0.5)
    dir_acc = 100 * np.mean((Y_test > 0) == (y_pred > 0))
    n_correct = np.sum((Y_test > 0) == (y_pred > 0))
    n_trials = len(Y_test)
    
    # Uji signifikansi binomial
    if n_trials > 0:
        binom_result = binomtest(n_correct, n_trials, p=0.5)
        p_value = binom_result.pvalue
        ci_low, ci_high = binom_result.proportion_ci(confidence_level=0.95)
        
        # Uji korelasi
        with np.errstate(invalid='ignore'):
            corr, corr_p_value = pearsonr(Y_test, y_pred)
        
        # Log warning jika perlu
        if p_value >= 0.05:
            logger.warning(f"Binomial p-value ({p_value:.4f}) not significant (<0.05)")
        if ci_low < 52 and n_trials >= 50:
            logger.warning(f"CI lower bound ({ci_low:.2f}%) below target (>52%)")
        if not np.isnan(corr_p_value) and corr_p_value >= 0.05:
            logger.warning(f"Pearson p-value ({corr_p_value:.4f}) not significant (<0.05)")
    else:
        p_value = np.nan
        ci_low = np.nan
        ci_high = np.nan
        corr = np.nan
        corr_p_value = np.nan
        logger.warning("No test samples for significance tests")
    
    logger.info(f"XGB Model - MAE: {mae:.6f}")
    logger.info(f"XGB Model - RMSE: {rmse:.6f}")
    logger.info(f"XGB Model - R² Score: {r2:.6f}")
    logger.info(f"XGB Model - ZPTAE Score: {zptae_score:.6f}")
    logger.info(f"XGB Model - Directional Accuracy: {dir_acc:.2f}%")
    
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_xgb_model_{prediction_horizon}m.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_xgb_scaler_{prediction_horizon}m.pkl')
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler disimpan dengan {scaler.n_features_in_} fitur di {scaler_path}")
    
    metrics = {
        'model': 'xgb_logreturn',
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'zptae': float(zptae_score),
        'directional_accuracy': float(dir_acc),
        'binom_p_value': float(p_value),
        'binom_ci_low': float(ci_low),
        'binom_ci_high': float(ci_high),
        'pearson_corr': float(corr),
        'pearson_p_value': float(corr_p_value)
    }
    
    metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_xgb_metrics_{prediction_horizon}m.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
        
    logger.info(f"Model XGBoost untuk {token_name} disimpan ke {model_path}")
    return metrics

def train_and_save_lgbm_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon):
    """
    Melatih dan menyimpan model LightGBM untuk prediksi log-return
    
    Args:
        token_name (str): Nama token (misalnya 'btcusd')
        X_train (numpy.array): Fitur pelatihan
        Y_train (numpy.array): Target pelatihan (log returns)
        X_test (numpy.array): Fitur pengujian
        Y_test (numpy.array): Target pengujian (log returns)
        prev_Y_test (numpy.array): Harga sebelumnya untuk perhitungan akurasi arah
        scaler (MinMaxScaler): Scaler yang telah di-fit
        prediction_horizon (int): Horizon prediksi dalam menit
        
    Returns:
        dict: Metrik untuk model yang dilatih
    """
    logger.info(f"Melatih model LightGBM untuk {token_name}")
    
    tscv = TimeSeriesSplit(n_splits=5)
    pipeline = Pipeline([('lgbm', LGBMRegressor(random_state=42, objective='regression', force_col_wise=True, verbosity=-1))])

    param_space = {
        'lgbm__n_estimators': [50, 150],
        'lgbm__max_depth': [5, 8, 13],
        'lgbm__learning_rate': [0.01, 0.1],
        'lgbm__subsample': [0.5, 1.0],
        'lgbm__colsample_bytree': [0.5, 1.0],
        'lgbm__min_child_weight': [1, 5],
        'lgbm__lambda_l1': [0, 3],
        'lgbm__lambda_l2': [0, 3],
        'lgbm__reg_alpha': [0, 5],
        'lgbm__bagging_freq': [1],
        'lgbm__num_leaves': [10, 18, 25],
        'lgbm__min_data_in_leaf': [1, 10]
    }
    
    search = BayesSearchCV(
        pipeline, param_space, n_iter=7, cv=tscv,
        scoring='neg_mean_absolute_error', n_jobs=3, random_state=42, verbose=0
    )

    # Pelatihan model
    logger.info("Starting BayesSearchCV fitting for LightGBM...")
    search.fit(X_train, Y_train)
    logger.info("BayesSearchCV fitting completed for LightGBM.")
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    zptae_score = calculate_zptae(Y_test, y_pred, alpha=0.5)
    dir_acc = 100 * np.mean((Y_test > 0) == (y_pred > 0))
    n_correct = np.sum((Y_test > 0) == (y_pred > 0))
    n_trials = len(Y_test)
    
    # Uji signifikansi binomial
    if n_trials > 0:
        binom_result = binomtest(n_correct, n_trials, p=0.5)
        p_value = binom_result.pvalue
        ci_low, ci_high = binom_result.proportion_ci(confidence_level=0.95)
        
        # Uji korelasi
        with np.errstate(invalid='ignore'):
            corr, corr_p_value = pearsonr(Y_test, y_pred)
        
        # Log warning jika perlu
        if p_value >= 0.05:
            logger.warning(f"Binomial p-value ({p_value:.4f}) not significant (<0.05)")
        if ci_low < 52 and n_trials >= 50:
            logger.warning(f"CI lower bound ({ci_low:.2f}%) below target (>52%)")
        if not np.isnan(corr_p_value) and corr_p_value >= 0.05:
            logger.warning(f"Pearson p-value ({corr_p_value:.4f}) not significant (<0.05)")
    else:
        p_value = np.nan
        ci_low = np.nan
        ci_high = np.nan
        corr = np.nan
        corr_p_value = np.nan
        logger.warning("No test samples for significance tests")

    logger.info(f"LGBM Model - MAE: {mae:.6f}")
    logger.info(f"LGBM Model - RMSE: {rmse:.6f}")
    logger.info(f"LGBM Model - R² Score: {r2:.6f}")
    logger.info(f"LGBM Model - ZPTAE Score: {zptae_score:.6f}")
    logger.info(f"LGBM Model - Directional Accuracy: {dir_acc:.2f}%")
    
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lgbm_model_{prediction_horizon}m.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lgbm_scaler_{prediction_horizon}m.pkl')
    
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler disimpan dengan {scaler.n_features_in_} fitur di {scaler_path}")
    
    metrics = {
        'model': 'lgbm_logreturn',
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'r2': float(r2),
        'zptae': float(zptae_score),
        'directional_accuracy': float(dir_acc),
        'binom_p_value': float(p_value),
        'binom_ci_low': float(ci_low),
        'binom_ci_high': float(ci_high),
        'pearson_corr': float(corr),
        'pearson_p_value': float(corr_p_value)
    }
    
    metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lgbm_metrics_{prediction_horizon}m.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
        
    logger.info(f"Model LightGBM untuk {token_name} disimpan ke {model_path}")
    return metrics

def train_log_return_models(token_name, look_back=60, prediction_horizon=480, hours_data=None):
    """
    Fungsi utama untuk melatih semua model dengan pipeline yang diperbaiki
    """
    try:
        logger.info(f"Melatih model Log-Return untuk {token_name} dengan horizon {prediction_horizon} steps")
        
        # 1. Ambil snapshot data terkini
        data_df = get_data_snapshot(token_name)
        if data_df is None:
            logger.error(f"Gagal memuat snapshot data untuk {token_name}")
            return None
            
        # 2. Jika diperlukan, filter berdasarkan jam
        if hours_data is not None:
            max_date = data_df['date'].max()
            cutoff_date = max_date - timedelta(hours=hours_data)
            data_df = data_df[data_df['date'] >= cutoff_date]
            logger.info(f"Memfilter snapshot data ke {hours_data} jam terakhir: {len(data_df)} baris")
            
        # Validasi jumlah data
        if len(data_df) < 5000:  # PERBAIKAN: Minimum data yang lebih tinggi
            logger.error(f"Data snapshot terlalu sedikit: {len(data_df)} records, butuh minimal 5000")
            return None
            
        # 3. Proses data dengan fungsi yang diperbaiki
        logger.info("Menyiapkan data untuk model dengan pipeline yang diperbaiki...")
        result = prepare_data_for_log_return(data_df, look_back, prediction_horizon)
        
        if result is None:
            logger.error("Gagal menyiapkan data untuk model")
            return None
        
        X_train, Y_train, prev_Y_train, X_test, Y_test, prev_Y_test, feature_columns = result
        
        # Validasi data
        if len(X_train) == 0 or len(X_test) == 0:
            logger.error("Data training atau testing kosong")
            return None
        
        # PERBAIKAN: Scaling setelah split dan hanya pada training data
        scaler_xgb = MinMaxScaler(feature_range=(-1, 1))  # Range yang lebih sesuai untuk returns
        
        # Fit hanya pada training data
        X_train_scaled = scaler_xgb.fit_transform(X_train)
        X_test_scaled = scaler_xgb.transform(X_test)
        
        logger.info(f"Ukuran set pelatihan: {len(X_train_scaled)}, Ukuran set pengujian: {len(X_test_scaled)}")
        logger.info(f"Distribusi log-return training: mean={np.nanmean(Y_train):.6f}, std={np.nanstd(Y_train):.6f}")
        logger.info(f"Distribusi log-return testing: mean={np.nanmean(Y_test):.6f}, std={np.nanstd(Y_test):.6f}")
        
        # Handle NaN values
        train_mask = ~np.isnan(Y_train)
        test_mask = ~np.isnan(Y_test)
        
        X_train_clean = X_train_scaled[train_mask]
        Y_train_clean = Y_train[train_mask]
        X_test_clean = X_test_scaled[test_mask]
        Y_test_clean = Y_test[test_mask]
        prev_Y_test_clean = prev_Y_test[test_mask]
        
        if len(X_train_clean) == 0 or len(X_test_clean) == 0:
            logger.error("Tidak ada data valid setelah cleaning NaN")
            return None
        
        results = {}
        
        # Train model XGB
        xgb_metrics = train_and_save_xgb_model(
            token_name, X_train_clean, Y_train_clean, X_test_clean, Y_test_clean, 
            prev_Y_test_clean, scaler_xgb, prediction_horizon
        )
        if xgb_metrics:
            results['xgb'] = xgb_metrics
        
        # Train model LGBM
        lgbm_metrics = train_and_save_lgbm_model(
            token_name, X_train_clean, Y_train_clean, X_test_clean, Y_test_clean,
            prev_Y_test_clean, scaler_xgb, prediction_horizon
        )
        if lgbm_metrics:
            results['lgbm'] = lgbm_metrics
        
        return results
    
    except Exception as e:
        logger.error(f"Error melatih model log-return untuk {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    logger.info("Memulai proses pelatihan kedua (2) model")
    
    try:
        # Setup direktori
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            logger.info(f"Membuat direktori model di {MODELS_DIR}")
        
        os.makedirs(TIINGO_DATA_DIR, exist_ok=True)
        logger.info(f"Direktori cache: Tiingo={TIINGO_DATA_DIR}")
        
        # Konfigurasi training
        tokens = ["btcusd"]
        look_back = 60
        prediction_horizon = 480
        hours_data = None  # Gunakan seluruh data snapshot
        
        all_results = {}
        for token_name in tokens:
            logger.info(f"\n{'='*50}")
            logger.info(f"Melatih model Log-Return untuk {token_name.upper()} dengan horizon {prediction_horizon} steps")
            logger.info(f"Jendela look back: {look_back} steps")
            logger.info(f"Menggunakan snapshot data terkini")
            logger.info(f"{'='*50}")
            
            # Eksekusi training dengan snapshot
            results = train_log_return_models(token_name, look_back, prediction_horizon, hours_data)
            
            if results:
                all_results[token_name] = results
                comparison_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_comparison_{prediction_horizon}m.json')
                with open(comparison_path, 'w') as f:
                    json.dump(results, f)
                
                best_model = min(results.items(), key=lambda x: x[1]['zptae'])[0]
                logger.info(f"Model terbaik untuk {token_name} log-return: {best_model} (ZPTAE: {results[best_model]['zptae']:.6f})")
                
                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 2, 1)
                models = list(results.keys())
                zptae_values = [results[model]['zptae'] for model in models]
                plt.bar(models, zptae_values, color="#1c60df")
                plt.title('Perbandingan ZPTAE (Lebih Rendah Lebih Baik)')
                plt.ylabel('ZPTAE')
                
                plt.subplot(2, 2, 2)
                mae_values = [results[model]['mae'] for model in models]
                plt.bar(models, mae_values, color="#3baa3b")
                plt.title('Perbandingan MAE (Lebih Rendah Lebih Baik)')
                plt.ylabel('MAE')
                
                plt.subplot(2, 2, 3)
                r2_values = [results[model]['r2'] for model in models]
                plt.bar(models, r2_values, color="#F5593EFF")
                plt.title('Perbandingan R² (Lebih Tinggi Lebih Baik)')
                plt.ylabel('R²')
                
                plt.subplot(2, 2, 4)
                dir_acc_values = [results[model]['directional_accuracy'] for model in models]
                plt.bar(models, dir_acc_values, color='#9149D4FF')
                plt.title('Perbandingan Akurasi Arah (Lebih Tinggi Lebih Baik)')
                plt.ylabel('Akurasi Arah (%)')
                
                plt.tight_layout()
                
                # Simpan plot dengan pengecekan direktori
                plot_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_comparison_{prediction_horizon}m.png')
                plt.savefig(plot_path)
                plt.close()
        
        logger.info("\nRingkasan Model yang sudah dilatih dan dipilih terbaik ZPTAE & Akurasi:")
        for token_name in tokens:
            if token_name in all_results:
                results = all_results[token_name]
                best_model = min(results.items(), key=lambda x: x[1]['zptae'])[0]
                logger.info(f"  {token_name.upper()} Log-Return ({prediction_horizon}m): {best_model}")
                logger.info(f"  ZPTAE: {results[best_model]['zptae']:.6f}")
                logger.info(f"  Akurasi Arah: {results[best_model]['directional_accuracy']:.2f}%")
            else:
                logger.info(f"  {token_name.upper()} Log-Return ({prediction_horizon}m): Tidak ada model yang dilatih")
        
    except Exception as e:
        logger.error(f"Error dalam proses pelatihan: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Proses pelatihan selesai. Exit container.")
        sys.exit(0)
