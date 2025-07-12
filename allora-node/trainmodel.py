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
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
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
        logging.FileHandler(f'paxg_model_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

# Import config untuk path
try:
    from config import DATA_BASE_PATH, TIINGO_DATA_DIR
except ImportError:
    logger.warning("Gagal mengimpor dari app_config. Mengatur path default.")
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
    Memuat data OHLCV dari file cache Tiingo JSON
    Args:
        token_name (str): Nama token (misalnya 'paxgusd')
        cache_dir (str): Direktori cache Tiingo
    Returns:
        pd.DataFrame: DataFrame dengan data OHLCV atau None jika gagal
    """
    base_ticker = token_name.replace('usd', '').lower()
    cache_path = os.path.join(cache_dir, "tiingo_data_5min.json")
    
    if not os.path.exists(cache_path):
        logger.error(f"Cache file not found: {cache_path}")
        return None
        
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
            
        df = pd.DataFrame(data['paxgusd']['priceData'])
        
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
    df.interpolate(method='linear', inplace=True)  # Pertahankan interpolasi untuk tren
    df.bfill(inplace=True)
    df.ffill(inplace=True)
    if df.isna().any().any():
        logger.error(f"Data snapshot mengandung NaN setelah pembersihan: {df.columns[df.isna().any()]}")
        return None

    logger.info(f"Snapshot data berhasil diambil: {len(df)} records")
    return df

def calculate_log_returns(prices, period=1440):
    """
    Menghitung log returns untuk harga yang diberikan dengan periode tertentu.
    Args:
        prices (numpy.array or pandas.Series): Data harga
        period (int): Periode untuk menghitung returns dalam data poin (default 1440 untuk 24 jam)
    Returns:
        numpy.array: Array log returns
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
    
    # log_returns = np.log(prices[period:] / prices[:-period])
    log_returns = np.log((prices[period:] + 1e-8) / (prices[:-period] + 1e-8))
    return log_returns

def calculate_zptae(y_true, y_pred, sigma=None, alpha=1.5, window_size=100):
    """
    Menghitung Z-transformed Power-Tanh Absolute Error (ZPTAE) dengan optimasi.
    Args:
        y_true (numpy.array): Nilai aktual
        y_pred (numpy.array): Nilai prediksi
        sigma (float, optional): Standar deviasi untuk normalisasi
        alpha (float): Parameter power-law untuk PowerTanh (default 1.5)
        window_size (int): Ukuran jendela untuk perhitungan std (default 100)
    Returns:
        float: Skor ZPTAE (lebih rendah lebih baik)
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    abs_errors = np.abs(y_true - y_pred)
    abs_errors = np.clip(abs_errors, 0, np.percentile(abs_errors, 95))  # Batasi outlier
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

def prepare_data_for_log_return(data_df, look_back=60, prediction_horizon=1440):
    """
    Menyiapkan data untuk pelatihan model prediksi log-return
    Perubahan utama:
        - Hapus proses scaling dari fungsi ini
        - Kembalikan data mentah (belum discale)
        - Scaler tidak dibuat di sini
        - Sesuaikan dengan fitur baru: volatility_range, price_momentum, price_trend, volumeNotional
    """
    logger.info(f"Menyiapkan data dengan look_back={look_back}, prediction_horizon={prediction_horizon}")
    
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Input harus berupa pandas DataFrame")
    
    required_cols = ['close', 'open', 'high', 'low', 'volume', 'volumeNotional']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Kolom yang diperlukan tidak ada: {missing_cols}")
    
    df = data_df.copy()
    price_array = df['close'].values
    
    # Validasi jumlah data
    min_required = prediction_horizon + look_back + 100
    if len(price_array) < min_required:
        logger.error(f"Data tidak cukup! Hanya {len(price_array)} records, butuh minimal {min_required}")
        return None, None, None, None
        
    all_log_returns = calculate_log_returns(price_array, period=prediction_horizon)
    
    # Hitung fitur teknikal dengan penanganan pembagian nol
    open_replaced = df['open'].replace(0, 1e-8)
    df['volatility_range'] = ((df['high'] - df['low']) / open_replaced) * 100  # Amplifikasi *100 untuk sensitivitas emas rendah var
    df['price_momentum'] = df['close'].diff().rolling(window=6, min_periods=1).mean().fillna(0) * 100  # Momentum 1 jam (12 × 5 menit)
    df['price_trend'] = df['close'].pct_change().rolling(window=6, min_periods=1).mean().fillna(0) * 100  # Tren 1 jam (12 × 5 menit)
        
    # Gunakan volumeNotional langsung tanpa normalisasi terpisah
    # df['volumeNotional'] = df['volumeNotional']  # Pertahankan sebagai fitur mentah
    # Opsional: Normalisasi rolling jika diperlukan (komentar untuk pengujian)
    df['volumeNotional'] = df['volumeNotional'] / df['volumeNotional'].rolling(window=20, min_periods=1).mean().replace(0, 1e-8)
    
    # Daftar fitur yang disesuaikan
    feature_columns = [
        'close', 'open', 'high', 'low',
        'volatility_range',
        'sma_10', 'dist_sma_10',
        'sma_cross_8_13', 'sma_cross_signal',
        'log_return_1h', 'log_return_4h',
        'volumeNotional',
        'price_momentum',
        'price_trend'
    ]
    
    # Hitung SMA dan jarak
    if len(df) > 10:
        df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['dist_sma_10'] = (df['close'] - df['sma_10']) / open_replaced
    
    # Hitung SMA cross 8-13 (disesuaikan dengan data yang ada)
    if 'sma_8' not in df.columns and len(df) > 8:
        df['sma_8'] = df['close'].rolling(window=8, min_periods=1).mean()
    if 'sma_13' not in df.columns and len(df) > 13:
        df['sma_13'] = df['close'].rolling(window=13, min_periods=1).mean()
    if 'sma_13' in df.columns:
        df['sma_cross_8_13'] = df['sma_8'] - df['sma_13']
        df['sma_cross_signal'] = np.where(df['sma_cross_8_13'] > 0, 1, np.where(df['sma_cross_8_13'] < 0, -1, 0))
    
    # Tambahkan log return cara hitung 1hx60/5menit
    df['log_return_1h'] = np.log(df['close'] / df['close'].shift(12)).fillna(0)
    df['log_return_4h'] = np.log(df['close'] / df['close'].shift(48)).fillna(0)
    
    # Penanganan nilai tak hingga dan NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.interpolate(method='linear').bfill().ffill()  # Ganti fillna(0) dengan interpolasi
    
    # Pastikan ada cukup data setelah penyesuaian
    if len(df) <= prediction_horizon:
        logger.error(f"Setelah pembersihan, data tidak cukup ({len(df)}) untuk horizon {prediction_horizon}")
        return None, None, None, None
        
    df_adjusted = df.iloc[:-prediction_horizon].copy()
    
    logger.info(f"Bentuk dataframe asli: {df.shape}")
    logger.info(f"Bentuk dataframe yang disesuaikan: {df_adjusted.shape}")
    logger.info(f"Bentuk log returns: {all_log_returns.shape}")
    
    # HAPUS PROSES SCALING DI SINI
    X, Y, prev_Y = [], [], []
    for i in range(len(df_adjusted) - look_back + 1):
        if i < len(all_log_returns):
            # Ambil jendela fitur sebagai DataFrame (data mentah)
            window_df = df_adjusted.iloc[i:i + look_back][feature_columns]  # Shape: (60, n_features)
            
            # Simpan data mentah (tanpa scaling)
            window_features = window_df.values.flatten()  # Shape: (60 * n_features,)
            X.append(window_features)
            Y.append(all_log_returns[i])
            prev_Y.append(window_df['close'].iloc[-1])  # Harga close terakhir (mentah)
    
    X = np.array(X)  # Shape: (n_samples, 60 * n_features)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    
    logger.info(f"Bentuk X setelah persiapan: {X.shape} (mentah)")
    logger.info(f"Bentuk Y setelah persiapan: {Y.shape}")
    logger.info(f"Bentuk prev_Y setelah persiapan: {prev_Y.shape}")
    
    # Kembalikan None untuk scaler (akan dibuat di tempat lain)
    return X, Y, prev_Y, None

def prepare_data_for_lstm(data_df, look_back=60, prediction_horizon=1440):
    """
    Menyiapkan data untuk model LSTM dengan bentuk 3D yang sesuai
    Perubahan utama:
        - Hapus proses scaling dari fungsi ini
        - Kembalikan data mentah (belum discale)
        - Scaler tidak dibuat di sini
        - Sesuaikan dengan fitur baru
    """
    logger.info(f"Menyiapkan data untuk LSTM dengan look_back={look_back}, prediction_horizon={prediction_horizon}")
    
    if not isinstance(data_df, pd.DataFrame):
        raise ValueError("Input harus berupa pandas DataFrame")
    
    required_cols = ['close', 'open', 'high', 'low', 'volume', 'volumeNotional']
    missing_cols = [col for col in required_cols if col not in data_df.columns]
    if missing_cols:
        raise ValueError(f"Kolom yang diperlukan tidak ada: {missing_cols}")
    
    df = data_df.copy()
    price_array = df['close'].values
    
    # Validasi jumlah data
    min_required = prediction_horizon + look_back + 100
    if len(price_array) < min_required:
        logger.error(f"Data tidak cukup! Hanya {len(price_array)} records, butuh minimal {min_required}")
        return None, None, None, None
        
    all_log_returns = calculate_log_returns(price_array, period=prediction_horizon)
    
    # Hitung fitur teknikal dengan penanganan pembagian nol
    open_replaced = df['open'].replace(0, 1e-8)
    df['volatility_range'] = ((df['high'] - df['low']) / open_replaced) * 100  # Amplifikasi *100 untuk sensitivitas emas rendah var
    df['price_momentum'] = df['close'].diff().rolling(window=6, min_periods=1).mean().fillna(0) * 100  # Momentum 1 jam (12 × 5 menit)
    df['price_trend'] = df['close'].pct_change().rolling(window=6, min_periods=1).mean().fillna(0) * 100  # Tren 1 jam (12 × 5 menit)
        
    # Gunakan volumeNotional langsung tanpa normalisasi terpisah
    # df['volumeNotional'] = df['volumeNotional']  # Pertahankan sebagai fitur mentah
    # Opsional: Normalisasi rolling jika diperlukan (komentar untuk pengujian)
    df['volumeNotional'] = df['volumeNotional'] / df['volumeNotional'].rolling(window=20, min_periods=1).mean().replace(0, 1e-8)
    
    # Daftar fitur yang disesuaikan
    feature_columns = [
        'close', 'open', 'high', 'low',
        'volatility_range',
        'sma_10', 'dist_sma_10',
        'sma_cross_8_13', 'sma_cross_signal',
        'log_return_1h', 'log_return_4h',
        'volumeNotional',
        'price_momentum',
        'price_trend'
    ]
    
    # Hitung SMA dan jarak
    if len(df) > 10:
        df['sma_10'] = df['close'].rolling(window=10, min_periods=1).mean()
        df['dist_sma_10'] = (df['close'] - df['sma_10']) / open_replaced
    
    # Hitung SMA cross 8-13 (disesuaikan dengan data yang ada)
    if 'sma_8' not in df.columns and len(df) > 8:
        df['sma_8'] = df['close'].rolling(window=8, min_periods=1).mean()
    if 'sma_13' not in df.columns and len(df) > 13:
        df['sma_13'] = df['close'].rolling(window=13, min_periods=1).mean()
    if 'sma_13' in df.columns:
        df['sma_cross_8_13'] = df['sma_8'] - df['sma_13']
        df['sma_cross_signal'] = np.where(df['sma_cross_8_13'] > 0, 1, np.where(df['sma_cross_8_13'] < 0, -1, 0))
    
    # Tambahkan log return cara hitung 1hx60/5menit
    df['log_return_1h'] = np.log(df['close'] / df['close'].shift(12)).fillna(0)
    df['log_return_4h'] = np.log(df['close'] / df['close'].shift(48)).fillna(0)
    
    # Penanganan nilai tak hingga dan NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.interpolate(method='linear').bfill().ffill()  # Ganti fillna(0) dengan interpolasi
    
    # Pastikan ada cukup data setelah penyesuaian
    if len(df) <= prediction_horizon:
        logger.error(f"Setelah pembersihan, data tidak cukup ({len(df)}) untuk horizon {prediction_horizon}")
        return None, None, None, None
        
    df_adjusted = df.iloc[:-prediction_horizon].copy()
    
    logger.info(f"Bentuk dataframe asli untuk LSTM: {df.shape}")
    logger.info(f"Bentuk dataframe yang disesuaikan untuk LSTM: {df_adjusted.shape}")
    
    # HAPUS PROSES SCALING DI SINI
    X, Y, prev_Y = [], [], []
    for i in range(len(df_adjusted) - look_back + 1):
        if i < len(all_log_returns):
            # Ambil jendela fitur sebagai DataFrame (data mentah)
            window_df = df_adjusted.iloc[i:i + look_back][feature_columns]  # Shape: (60, n_features)
            
            # Simpan data mentah (tanpa scaling)
            X.append(window_df.values)
            Y.append(all_log_returns[i])
            prev_Y.append(window_df['close'].iloc[-1])  # Harga close terakhir (mentah)
    
    X = np.array(X)  # Shape: (n_samples, 60, n_features)
    Y = np.array(Y)
    prev_Y = np.array(prev_Y)
    
    logger.info(f"Bentuk X LSTM: {X.shape} (mentah)")
    logger.info(f"Bentuk Y LSTM: {Y.shape}")
    
    # Kembalikan None untuk scaler (akan dibuat di tempat lain)
    return X, Y, prev_Y, None

def train_and_save_xgb_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon):
    """
    Melatih dan menyimpan model XGBoost untuk prediksi log-return
    
    Args:
        token_name (str): Nama token (misalnya 'paxgusd')
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
    
    tscv = TimeSeriesSplit(n_splits=4)
    pipeline = Pipeline([('xgb', XGBRegressor(random_state=42, objective='reg:squarederror', verbosity=0))])

    param_space = {
        'xgb__n_estimators': [50, 200],
        'xgb__max_depth': [5, 8, 13],
        'xgb__learning_rate': [0.01, 0.1],
        'xgb__subsample': [0.5, 1.0],
        'xgb__colsample_bytree': [0.5, 1.0],
        'xgb__min_child_weight': [1, 7],
        'xgb__max_delta_step': [0, 5],
        'xgb__gamma': [0, 1],
        'xgb__reg_lambda': [0, 2],
        'xgb__reg_alpha': [0, 2],
        'xgb__booster': ['gbtree', 'gblinear']
    }
    
    search = BayesSearchCV(
        pipeline, param_space, n_iter=10, cv=tscv,
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
    zptae_score = calculate_zptae(Y_test, y_pred, alpha=1.5)
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
        token_name (str): Nama token (misalnya 'paxgusd')
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
    
    tscv = TimeSeriesSplit(n_splits=4)
    pipeline = Pipeline([('lgbm', LGBMRegressor(random_state=42, objective='regression', force_col_wise=True, verbosity=-1))])

    param_space = {
        'lgbm__n_estimators': [50, 200],
        'lgbm__max_depth': [5, 8, 13],
        'lgbm__learning_rate': [0.01, 0.1],
        'lgbm__subsample': [0.5, 1.0],
        'lgbm__colsample_bytree': [0.5, 1.0],
        'lgbm__min_child_weight': [1, 5],
        'lgbm__lambda_l1': [0, 2],
        'lgbm__lambda_l2': [0, 2],
        'lgbm__reg_alpha': [0, 2],
        'lgbm__bagging_freq': [1],
        'lgbm__num_leaves': [10, 20, 30],
        'lgbm__min_data_in_leaf': [1, 10]
    }
    
    search = BayesSearchCV(
        pipeline, param_space, n_iter=10, cv=tscv,
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
    zptae_score = calculate_zptae(Y_test, y_pred, alpha=1.5)
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

def train_and_save_lstm_model(token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler, prediction_horizon, look_back):
    """
    Melatih dan menyimpan model LSTM untuk prediksi log-return
    Args:
        token_name (str): Nama token (misalnya 'paxgusd')
        X_train (numpy.array): Fitur pelatihan (bentuk 3D untuk LSTM)
        Y_train (numpy.array): Target pelatihan (log returns)
        X_test (numpy.array): Fitur pengujian (bentuk 3D untuk LSTM)
        Y_test (numpy.array): Target pengujian (log returns)
        prev_Y_test (numpy.array): Harga sebelumnya untuk perhitungan akurasi arah
        scaler (MinMaxScaler): Scaler yang telah di-fit
        prediction_horizon (int): Horizon prediksi dalam menit
        look_back (int): Ukuran jendela look back
    Returns:
        dict: Metrik untuk model yang dilatih
    """
    logger.info(f"Melatih model LSTM untuk {token_name}")
    
    # Validasi ukuran data
    if len(X_train) < 100:
        logger.warning(f"Data training LSTM sangat kecil: {len(X_train)} samples")
    
    # Split validation set hanya jika data cukup
    val_split = 0.2
    val_size = int(val_split * len(X_train))
    if val_size < 10:
        logger.warning("Menggunakan semua data untuk training karena data kecil")
        X_train_split = X_train
        Y_train_split = Y_train
        X_val = None
        Y_val = None
    else:
        split_index = int((1 - val_split) * len(X_train))
        X_train_split, X_val = X_train[:split_index], X_train[split_index:]
        Y_train_split, Y_val = Y_train[:split_index], Y_train[split_index:]
    
    n_features = X_train.shape[2]

    # Model LSTM yang disederhanakan
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=False, input_shape=(look_back, n_features)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(units=64))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    
    optimizer = Adam(learning_rate=0.0005)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    
    # Sesuaikan callbacks berdasarkan ketersediaan data validasi
    callbacks = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=6, min_lr=0.00005)]
    if X_val is not None:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))
    
    history = model.fit(
        X_train_split, Y_train_split,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, Y_val) if X_val is not None else None,
        callbacks=callbacks,
        verbose=0
    )
    
    y_pred = model.predict(X_test, verbose=0).flatten()
    y_pred = smooth_predictions(y_pred, alpha=0.1)
    
    mae = mean_absolute_error(Y_test, y_pred)
    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)
    zptae_score = calculate_zptae(Y_test, y_pred, alpha=1.5)
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
    
    logger.info(f"LSTM Model - MAE: {mae:.6f}")
    logger.info(f"LSTM Model - RMSE: {rmse:.6f}")
    logger.info(f"LSTM Model - R² Score: {r2:.6f}")
    logger.info(f"LSTM Model - ZPTAE Score: {zptae_score:.6f}")
    logger.info(f"LSTM Model - Directional Accuracy: {dir_acc:.2f}%")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'LSTM Log-Return Model Loss untuk {token_name} ({prediction_horizon}m)')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    # Pastikan direktori ada sebelum menyimpan plot
    os.makedirs(MODELS_DIR, exist_ok=True)
    loss_plot_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lstm_loss_{prediction_horizon}m.png')
    plt.savefig(loss_plot_path)
    plt.close()
    
    model_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lstm_model_{prediction_horizon}m.keras')
    scaler_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lstm_scaler_{prediction_horizon}m.pkl')
    
    save_model(model, model_path)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler disimpan dengan {scaler.n_features_in_} fitur di {scaler_path}")
    
    metrics = {
        'model': 'lstm_logreturn',
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
    
    metrics_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_lstm_metrics_{prediction_horizon}m.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
        
    logger.info(f"Model LSTM untuk {token_name} disimpan ke {model_path}")
    return metrics

def train_log_return_models(token_name, look_back=60, prediction_horizon=1440, hours_data=None):
    """
    Fungsi utama untuk melatih semua model dengan snapshot data
    """
    try:
        logger.info(f"Melatih model Log-Return untuk {token_name} dengan horizon {prediction_horizon} menit")
        
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
        if len(data_df) < 1000:
            logger.error(f"Data snapshot terlalu sedikit: {len(data_df)} records")
            return None
            
        results = {}
        
        # 3. Proses data untuk model XGB dan LGBM menggunakan snapshot
        logger.info("Menyiapkan data untuk model XGBoost dan LightGBM...")
        X, Y, prev_Y, _ = prepare_data_for_log_return(data_df, look_back, prediction_horizon)
        
        if X is None or len(X) == 0:
            logger.error("Gagal menyiapkan data untuk model XGB/LGBM")
            return None
        
        # Time-series split
        test_size = int(0.2 * len(X))
        if test_size == 0:
            logger.error("Data tidak cukup untuk pembagian train/test")
            return None
            
        X_train_raw, X_test_raw = X[:-test_size], X[-test_size:]
        Y_train, Y_test = Y[:-test_size], Y[-test_size:]
        prev_Y_test = prev_Y[-test_size:]
        
        # Scaling setelah split data
        scaler_xgb = MinMaxScaler(feature_range=(0, 1))
        X_train = scaler_xgb.fit_transform(X_train_raw)
        X_test = scaler_xgb.transform(X_test_raw)
        
        logger.info(f"Ukuran set pelatihan: {len(X_train)}, Ukuran set pengujian: {len(X_test)}")
        logger.info(f"Distribusi log-return set pengujian: mean={np.mean(Y_test):.4f}, std={np.std(Y_test):.4f}")
        if np.all(Y_test > 0) or np.all(Y_test < 0):
            logger.warning("Log-return set pengujian semuanya positif atau negatif, dapat membiaskan akurasi arah")
        
        # Train model XGB
        xgb_metrics = train_and_save_xgb_model(
            token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler_xgb, prediction_horizon
        )
        if xgb_metrics:
            results['xgb'] = xgb_metrics
        
        # Train model LGBM
        lgbm_metrics = train_and_save_lgbm_model(
            token_name, X_train, Y_train, X_test, Y_test, prev_Y_test, scaler_xgb, prediction_horizon
        )
        if lgbm_metrics:
            results['lgbm'] = lgbm_metrics
        
        # 4. Proses data untuk LSTM menggunakan snapshot yang sama
        logger.info("Menyiapkan data untuk model LSTM...")
        X_lstm, Y_lstm, prev_Y_lstm, _ = prepare_data_for_lstm(data_df, look_back, prediction_horizon)
        
        if X_lstm is None:
            logger.error("Gagal menyiapkan data untuk model LSTM")
        else:
            # Bagi data
            split_index = int(0.8 * len(X_lstm))
            if split_index <= 0:
                logger.error("Data LSTM tidak cukup untuk dibagi")
            else:
                X_train_raw_lstm = X_lstm[:split_index]
                X_test_raw_lstm = X_lstm[split_index:]
                Y_train_lstm = Y_lstm[:split_index]
                Y_test_lstm = Y_lstm[split_index:]
                prev_Y_test_lstm = prev_Y_lstm[split_index:]
                
                # Scaling khusus untuk LSTM
                n_train, look_back, n_features = X_train_raw_lstm.shape
                X_train_2d = X_train_raw_lstm.reshape(-1, n_features)
                
                scaler_lstm = MinMaxScaler(feature_range=(0, 1))
                X_train_scaled_2d = scaler_lstm.fit_transform(X_train_2d)
                X_train_lstm = X_train_scaled_2d.reshape(n_train, look_back, n_features)
                
                n_test = X_test_raw_lstm.shape[0]
                X_test_2d = X_test_raw_lstm.reshape(-1, n_features)
                X_test_scaled_2d = scaler_lstm.transform(X_test_2d)
                X_test_lstm = X_test_scaled_2d.reshape(n_test, look_back, n_features)

                logger.info(f"Ukuran set pelatihan LSTM: {len(X_train_lstm)}, Ukuran set pengujian: {len(X_test_lstm)}")
                logger.info(f"Distribusi log-return set pengujian (LSTM): mean={np.mean(Y_test_lstm):.4f}, std={np.std(Y_test_lstm):.4f}")
                if np.all(Y_test_lstm > 0) or np.all(Y_test_lstm < 0):
                    logger.warning("Log-return set pengujian (LSTM) semuanya positif atau negatif, dapat membiaskan akurasi arah")
                
                # Train model LSTM
                lstm_metrics = train_and_save_lstm_model(
                    token_name, 
                    X_train_lstm, 
                    Y_train_lstm, 
                    X_test_lstm, 
                    Y_test_lstm, 
                    prev_Y_test_lstm, 
                    scaler_lstm, 
                    prediction_horizon, 
                    look_back
                )
                if lstm_metrics:
                    results['lstm'] = lstm_metrics
        
        return results
    
    except Exception as e:
        logger.error(f"Error melatih model log-return untuk {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    logger.info("Memulai proses pelatihan ketiga (3) model")
    
    try:
        # Setup direktori
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
            logger.info(f"Membuat direktori model di {MODELS_DIR}")
        
        os.makedirs(TIINGO_DATA_DIR, exist_ok=True)
        logger.info(f"Direktori cache: Tiingo={TIINGO_DATA_DIR}")
        
        # Konfigurasi training
        token_name = "paxgusd"
        look_back = 60
        prediction_horizon = 1440
        hours_data = None  # Gunakan seluruh data snapshot
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Melatih model Log-Return untuk {token_name.upper()} dengan horizon {prediction_horizon} menit")
        logger.info(f"Jendela look back: {look_back} menit")
        logger.info(f"Menggunakan snapshot data terkini")
        logger.info(f"{'='*50}")
        
        # Eksekusi training dengan snapshot
        paxg_results = train_log_return_models(token_name, look_back, prediction_horizon, hours_data)
        
        if paxg_results:
            # Pastikan direktori ada sebelum menyimpan file
            os.makedirs(MODELS_DIR, exist_ok=True)
            
            comparison_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_comparison_{prediction_horizon}m.json')
            with open(comparison_path, 'w') as f:
                json.dump(paxg_results, f)
            
            best_paxg_model = min(paxg_results.items(), key=lambda x: x[1]['zptae'])[0]
            logger.info(f"Model terbaik untuk {token_name} log-return: {best_paxg_model} (ZPTAE: {paxg_results[best_paxg_model]['zptae']:.6f})")
            
            plt.figure(figsize=(15, 10))
            
            plt.subplot(2, 2, 1)
            models = list(paxg_results.keys())
            zptae_values = [paxg_results[model]['zptae'] for model in models]
            plt.bar(models, zptae_values, color="#1c60df")
            plt.title('Perbandingan ZPTAE (Lebih Rendah Lebih Baik)')
            plt.ylabel('ZPTAE')
            
            plt.subplot(2, 2, 2)
            mae_values = [paxg_results[model]['mae'] for model in models]
            plt.bar(models, mae_values, color="#3baa3b")
            plt.title('Perbandingan MAE (Lebih Rendah Lebih Baik)')
            plt.ylabel('MAE')
            
            plt.subplot(2, 2, 3)
            r2_values = [paxg_results[model]['r2'] for model in models]
            plt.bar(models, r2_values, color='#F76046FF')
            plt.title('Perbandingan R² (Lebih Tinggi Lebih Baik)')
            plt.ylabel('R²')
            
            plt.subplot(2, 2, 4)
            dir_acc_values = [paxg_results[model]['directional_accuracy'] for model in models]
            plt.bar(models, dir_acc_values, color='#9149D4FF')
            plt.title('Perbandingan Akurasi Arah (Lebih Tinggi Lebih Baik)')
            plt.ylabel('Akurasi Arah (%)')
            
            plt.tight_layout()
            
            # Simpan plot dengan pengecekan direktori
            plot_path = os.path.join(MODELS_DIR, f'{token_name.lower()}_logreturn_comparison_{prediction_horizon}m.png')
            plt.savefig(plot_path)
            plt.close()
        
        logger.info("\nRingkasan Model yang sudah dilatih dan dipilih terbaik ZPTAE & Akurasi:")
        if paxg_results:
            logger.info(f"  PAXG/USD Log-Return ({prediction_horizon}m): {best_paxg_model}")
            logger.info(f"  ZPTAE: {paxg_results[best_paxg_model]['zptae']:.6f}")
            logger.info(f"  Akurasi Arah: {paxg_results[best_paxg_model]['directional_accuracy']:.2f}%")
        else:
            logger.info(f"PAXG/USD Log-Return ({prediction_horizon}m): Tidak ada model yang dilatih")
        
    except Exception as e:
        logger.error(f"Error melatih model log-return untuk {token_name}: {e}")
        logger.error(traceback.format_exc())
    finally:
        logger.info("Proses pelatihan selesai. Exit container.")
        sys.exit(0)
