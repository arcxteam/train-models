import os
import logging
import json
import numpy as np
import joblib
import asyncio
import pandas as pd
import traceback
import time
from datetime import datetime
from flask import Flask, Response, jsonify
from utils import load_tiingo_data, prepare_features_for_prediction, create_features_for_inference, get_coingecko_prices

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('prediction_api.log')]
)
logger = logging.getLogger(__name__)

# Import app_config
try:
    from config import DATA_BASE_PATH, TIINGO_API_TOKEN, TIINGO_DATA_DIR, TIINGO_CACHE_TTL
except ImportError:
    logger.warning("Gagal mengimpor dari app_config. Mengatur path default.")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_BASE_PATH = os.path.join(BASE_DIR, 'data')
    TIINGO_API_TOKEN = os.environ.get('TIINGO_API_TOKEN', '')
    TIINGO_DATA_DIR = os.path.join(DATA_BASE_PATH, 'tiingo_data')
    TIINGO_CACHE_TTL = 150

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

if os.environ.get('MODELS_DIR'):
    MODELS_DIR = os.environ.get('MODELS_DIR')
    logger.info(f"Menggunakan direktori model dari environment: {MODELS_DIR}")
elif not os.path.exists(MODELS_DIR) and os.path.exists('/root/forge/allora/models/'):
    MODELS_DIR = '/root/forge/allora/models/'
    logger.info(f"Menggunakan direktori model alternatif: {MODELS_DIR}")

API_PORT = int(os.environ.get('API_PORT', 8000))
LOOK_BACK = int(os.environ.get('LOOK_BACK', 60))
PREDICTION_HORIZON = int(os.environ.get('PREDICTION_HORIZON', 480))

TOKEN_INFO = {
    'btcusd': {
        'full_name': 'BTC',
        'model_type': 'logreturn',
        'prediction_horizon': 480,
        'look_back': 60,
        'coingecko_id': 'bitcoin'
    }
}

HTTP_RESPONSE_CODE_200 = 200
HTTP_RESPONSE_CODE_404 = 404
HTTP_RESPONSE_CODE_500 = 500

MODEL_SELECTION_CACHE = {}
PREDICTION_CACHE = {}
PREDICTION_PERFORMANCE = {}

os.makedirs(TIINGO_DATA_DIR, exist_ok=True)

def load_model_comparison(token_name, prediction_horizon):
    """Memuat data perbandingan model dari file JSON"""
    json_path = os.path.join(MODELS_DIR, f"{token_name}_logreturn_comparison_{prediction_horizon}m.json")
    
    try:
        if not os.path.exists(json_path):
            logger.warning(f"File perbandingan tidak ditemukan: {json_path}")
            return None
        with open(json_path, 'r') as f:
            comparison_data = json.load(f)
        logger.info(f"Berhasil memuat data perbandingan dari {json_path}")
        return comparison_data
    except Exception as e:
        logger.error(f"Error memuat file perbandingan {json_path}: {e}")
        return None

def get_best_model(comparison_data):
    """
    Mendapatkan model terbaik berdasarkan ZPTAE
    Args:
        comparison_data (dict): Dictionary dengan data perbandingan model
    Returns:
        str: Tipe model terbaik ('lgbm', atau 'xgb')
    """
    if not comparison_data:
        logger.warning("Tidak ada data perbandingan, menggunakan default LightGBM")
        return 'lgbm'
    
    logger.info(f"Model yang tersedia: {list(comparison_data.keys())}")
    best_model = min(comparison_data.keys(), key=lambda x: comparison_data[x].get('zptae', float('inf')))
    zptae = comparison_data[best_model].get('zptae', 'N/A')
    dir_acc = comparison_data[best_model].get('directional_accuracy', 'N/A')
    logger.info(f"Memilih model terbaik berdasarkan ZPTAE: {best_model} (ZPTAE: {zptae:.6f}, Akurasi Arah: {dir_acc:.2f}%)")
    return best_model

def determine_best_model(token_name, prediction_horizon):
    """
    Menentukan model terbaik untuk token berdasarkan metrik
    Args:
        token_name (str): Nama token (huruf kecil, misalnya 'btcusd')
        prediction_horizon (int): Horizon prediksi dalam menit
    Returns:
        str: Tipe model ('lgbm', atau 'xgb')
    """
    cache_key = f"{token_name}_{prediction_horizon}"
    json_path = os.path.join(MODELS_DIR, f"{token_name}_logreturn_comparison_{prediction_horizon}m.json")
    
    if cache_key in MODEL_SELECTION_CACHE:
        cache_mtime = MODEL_SELECTION_CACHE[cache_key].get('mtime', 0)
        file_mtime = os.path.getmtime(json_path) if os.path.exists(json_path) else 0
        if cache_mtime >= file_mtime:
            logger.info(f"Menggunakan cache pemilihan model untuk {token_name} ({prediction_horizon}m): {MODEL_SELECTION_CACHE[cache_key]['model']}")
            return MODEL_SELECTION_CACHE[cache_key]['model']
    
    comparison_data = load_model_comparison(token_name, prediction_horizon)
    best_model = get_best_model(comparison_data) if comparison_data else 'lgbm'
    MODEL_SELECTION_CACHE[cache_key] = {
        'model': best_model,
        'mtime': os.path.getmtime(json_path) if os.path.exists(json_path) else time.time()
    }
    return best_model

def load_model_and_scaler(token_name, prediction_horizon):
    """
    Memuat model dan scaler yang sesuai
    
    Args:
        token_name (str): Nama token (huruf kecil, misalnya 'btcusd')
        prediction_horizon (int): Horizon prediksi dalam menit
        
    Returns:
        tuple: (model, scaler, model_type)
    """
    best_model_type = determine_best_model(token_name, prediction_horizon)
    model_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_{best_model_type}_model_{prediction_horizon}m.pkl')
    scaler_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_{best_model_type}_scaler_{prediction_horizon}m.pkl')
    
    logger.info(f"Mencoba memuat model {best_model_type} dari: {model_path}")
    logger.info(f"Path scaler: {scaler_path}")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler mengharapkan {scaler.n_features_in_} fitur")
            logger.info(f"Berhasil memuat model {best_model_type.upper()} untuk {token_name}")
            return model, scaler, best_model_type
        except Exception as e:
            logger.error(f"Error memuat model {best_model_type} untuk {token_name}: {e}")
    
    fallback_order = ['lgbm', 'xgb'] if best_model_type != 'lgbm' else ['xgb']
    for model_type in fallback_order:
        model_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_{model_type}_model_{prediction_horizon}m.pkl')
        scaler_path = os.path.join(MODELS_DIR, f'{token_name}_logreturn_{model_type}_scaler_{prediction_horizon}m.pkl')
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                logger.info(f"Memuat model fallback {model_type.upper()} untuk {token_name}")
                logger.info(f"Scaler mengharapkan {scaler.n_features_in_} fitur")
                return model, scaler, model_type
            except Exception as e:
                logger.error(f"Error memuat model fallback {model_type} untuk {token_name}: {e}")
    
    logger.error(f"Tidak ada model yang berfungsi untuk {token_name} dengan horizon {prediction_horizon}m")
    return None, None, None

def track_prediction(token_name, predicted_price, latest_price, timestamp):
    """
    Melacak prediksi untuk verifikasi nanti
    
    Args:
        token_name (str): Nama token
        predicted_price (float): Harga prediksi
        latest_price (float): Harga saat ini
        timestamp: Timestamp prediksi
    """
    global PREDICTION_PERFORMANCE
    if token_name not in PREDICTION_PERFORMANCE:
        PREDICTION_PERFORMANCE[token_name] = []
    
    PREDICTION_PERFORMANCE[token_name].append({
        'timestamp': int(timestamp),
        'prediction_time': time.time(),
        'predicted_price': float(predicted_price),
        'latest_price': float(latest_price),
        'prediction_delta': float(predicted_price - latest_price),
        'prediction_pct': float(((predicted_price - latest_price) / latest_price) * 100)
    })
    if len(PREDICTION_PERFORMANCE[token_name]) > 100:
        PREDICTION_PERFORMANCE[token_name] = PREDICTION_PERFORMANCE[token_name][-100:]

def cached_prediction(token_name, prediction_horizon):
    """
    Menghasilkan prediksi harga dengan pipeline yang diperbaiki
    Hanya untuk internal tracking, bukan untuk blockchain
    """
    cache_key = f"{token_name}_{prediction_horizon}"
    if cache_key in PREDICTION_CACHE:
        cache_age = time.time() - PREDICTION_CACHE[cache_key]['timestamp']
        if cache_age < TIINGO_CACHE_TTL:
            logger.info(f"Menggunakan prediksi cache untuk {token_name} ({cache_age:.1f} detik umur cache)")
            return PREDICTION_CACHE[cache_key]['prediction']
    
    start_time = time.time()
    logger.info(f"Memulai prediksi untuk {token_name} (horizon: {prediction_horizon}m)")
    
    token_config = TOKEN_INFO.get(token_name, {'look_back': LOOK_BACK, 'model_type': 'logreturn', 'prediction_horizon': prediction_horizon})
    look_back = token_config.get('look_back', LOOK_BACK)
    
    model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
    model_load_time = time.time() - start_time
    
    if model is None or scaler is None:
        logger.error(f"Gagal memuat model untuk {token_name}")
        return None
    
    logger.info(f"Memuat model {model_type.upper()} dalam {model_load_time:.3f}s")
    
    # Gunakan hanya data Tiingo untuk konsistensi
    data = load_tiingo_data(token_name)
    if data is None or data[0] is None or len(data[0]) < look_back:
        logger.error(f"Data tidak cukup untuk prediksi {token_name}: perlu {look_back}")
        return None
    
    df, _ = data
    
    # Gunakan harga Tiingo terakhir untuk konsistensi
    latest_price = float(df['close'].iloc[-1])
    latest_timestamp = int(df['timestamp'].iloc[-1])
    
    logger.info(f"Data terbaru dari Tiingo: {df['date'].iloc[-1]}, Harga terakhir: {latest_price:.4f}")
    
    # Siapkan fitur dengan data Tiingo
    df_pred = df.tail(look_back).copy()
    
    # Gunakan feature engineering yang konsisten
    df_pred = create_features_for_inference(df_pred)
    
    preprocess_start = time.time()
    X_pred = prepare_features_for_prediction(df_pred, look_back, scaler, model_type=model_type)
    
    if X_pred is None:
        logger.error(f"Gagal menyiapkan fitur untuk {model_type}")
        return None
    
    logger.info(f"Bentuk input {model_type.upper()}: {X_pred.shape}")
    
    try:
        predict_start = time.time()
        pred = model.predict(X_pred)
        predict_time = time.time() - predict_start
        
        # Handle prediction output
        if isinstance(pred, (np.ndarray, list)):
            log_return = float(pred[0]) if len(pred) > 0 else 0.0
        else:
            log_return = float(pred)
        
        # Validasi log return yang reasonable untuk internal tracking
        if abs(log_return) > 0.5:
            logger.warning(f"Log return prediksi tidak realistic: {log_return}, clamping")
            log_return = np.clip(log_return, -0.1, 0.1)
        
        prediction_value = latest_price * np.exp(log_return)
        
        # Lacak prediksi
        track_prediction(token_name, prediction_value, latest_price, latest_timestamp)
        
        total_prediction_time = time.time() - start_time
        
        logger.info(f"Prediksi internal untuk {token_name.upper()}: Log Return={log_return}, Harga=${prediction_value:.4f}")
        
        PREDICTION_CACHE[cache_key] = {
            'prediction': float(prediction_value),
            'timestamp': time.time(),
            'log_return': float(log_return),
            'latest_price': float(latest_price),
            'model_type': model_type,
            'data_source': 'Tiingo'
        }
        
        return prediction_value
        
    except Exception as e:
        logger.error(f"Error selama prediksi untuk {token_name}: {e}")
        logger.error(traceback.format_exc())
        return None

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Price Prediction API", HTTP_RESPONSE_CODE_200

@app.route('/health', methods=['GET'])
async def health_check():
    """Endpoint sederhana untuk pemeriksaan kesehatan"""
    return "OK", HTTP_RESPONSE_CODE_200

@app.route('/predict_log_return/<token>', methods=['GET'])
def predict_log_return(token):
    """
    Prediksi endpoint utama untuk inference ke jaringan Allora testnet.
    Kembalikan nilai log-return sebagai teks biasa.
    """
    logger.info(f"Raw token received: '{token}' (length: {len(token)})")

    token_upper = token.strip().upper()
    token_map = {'BTC': 'btcusd'}
    if token_upper not in token_map:
        logger.error(f"Invalid token: '{token}'")
        return Response(json.dumps({"error": f"Token {token} tidak didukung"}), status=404, mimetype='application/json')

    token_name = token_map[token_upper]
    prediction_horizon = TOKEN_INFO[token_name]['prediction_horizon']
    look_back = TOKEN_INFO[token_name]['look_back']

    try:
        # Muat data dari Tiingo
        data = load_tiingo_data(token_name)
        if data is None or data[0] is None:
            logger.error("Gagal memuat data dari Tiingo")
            return Response(json.dumps({"error": "Data tidak tersedia"}), status=404, mimetype='application/json')
        
        df, _ = data
        logger.info(f"Data setelah load_tiingo_data: {len(df)} records")

        if df is None or len(df) < look_back:
            logger.error(f"Data tidak cukup: perlu {look_back}, dapat {len(df) if df is not None else 0}")
            return Response(json.dumps({"error": "Data tidak cukup"}), status=404, mimetype='application/json')

        # Ambil look back baris terakhir untuk inferensi 
        df_pred = df.tail(look_back).copy()
        logger.info(f"Data untuk prediksi: {len(df_pred)} records (look_back: {look_back})")

        # Muat model dan scaler
        model, scaler, model_type = load_model_and_scaler(token_name, prediction_horizon)
        if model is None or scaler is None:
            logger.error("Model atau scaler tidak tersedia")
            return Response(json.dumps({"error": "Model prediksi tidak tersedia"}), status=500, mimetype='application/json')

        # Gunakan fungsi feature engineering yang konsisten
        df_pred = create_features_for_inference(df_pred)
        
        # Siapkan fitur untuk prediksi
        X_pred = prepare_features_for_prediction(df_pred, look_back, scaler, model_type=model_type)
        if X_pred is None:
            logger.error("Gagal menyiapkan fitur")
            return Response(json.dumps({"error": "Gagal mempersiapkan fitur"}), status=500, mimetype='application/json')

        # Periksa kecocokan fitur berdasarkan tipe model
        logger.info(f"Input shape: {X_pred.shape}, expected features: {scaler.n_features_in_}")
        if X_pred.shape[1] != scaler.n_features_in_:
            logger.error(f"Feature mismatch: expected {scaler.n_features_in_}, got {X_pred.shape[1]}")
            return Response(json.dumps({"error": "Feature mismatch"}), status=500, mimetype='application/json')

        # Lakukan prediksi log return
        pred = model.predict(X_pred)
        
        # Konversi prediksi menjadi float tunggal
        if isinstance(pred, (np.ndarray, list)):
            if len(pred) > 0:
                if isinstance(pred[0], (np.ndarray, list)):
                    log_return = float(pred[0][0])
                else:
                    log_return = float(pred[0])
            else:
                log_return = 0.0
        else:
            log_return = float(pred)

        # Gunakan harga Tiingo untuk konsistensi
        latest_price = float(df['close'].iloc[-1])
        latest_timestamp = int(df['timestamp'].iloc[-1])
        
        # Hitung predicted price untuk tracking
        predicted_price = latest_price * np.exp(log_return)

        # Log hasil prediksi
        logger.info(f"Prediksi selesai - Token: {token_upper}, Log-return: {log_return}, "
                    f"Harga saat ini: ${latest_price:.4f}, Harga prediksi: ${predicted_price:.4f}")

        # Simpan ke cache
        cache_key = f"{token_name}_{prediction_horizon}"
        PREDICTION_CACHE[cache_key] = {
            'timestamp': time.time(),
            'log_return': float(log_return),
            'latest_price': float(latest_price),
            'prediction': float(predicted_price),
            'data_source': 'Tiingo',
            'model_type': model_type
        }

        # Lacak prediksi
        track_prediction(token_name, predicted_price, latest_price, latest_timestamp)

        # Kembalikan nilai log-return ASLI tanpa formatting untuk blockchain
        logger.info(f"Mengembalikan log-return ke blockchain: {log_return}")
        
        return Response(str(log_return), status=200, mimetype='text/plain')

    except Exception as e:
        logger.error(f"Error memproses prediksi log-return: {e}\n{traceback.format_exc()}")
        return Response(json.dumps({"error": str(e)}), status=500, mimetype='application/json')

@app.route('/performance/<token>', methods=['GET'])
async def performance(token):
    """Mengembalikan metrik performa dan informasi model untuk token"""
    try:
        token_upper = token.strip().upper()
        token_map = {'BTC': 'btcusd'}
        if token_upper not in token_map:
            return jsonify({"status": "error", "message": f"Token {token} tidak didukung"})

        token_name = token_map[token_upper]
        prediction_horizon = TOKEN_INFO[token_name]['prediction_horizon']
        
        # Muat data perbandingan model
        comparison_data = load_model_comparison(token_name, prediction_horizon)
        best_model = determine_best_model(token_name, prediction_horizon)
        
        # Muat data cache prediksi
        cache_key = f"{token_name}_{prediction_horizon}"
        cache_data = PREDICTION_CACHE.get(cache_key, {})
        
        # Hitung metrik performa
        verified_predictions = [p for p in PREDICTION_PERFORMANCE.get(token_name, []) if 'actual_price' in p]
        accuracy_metrics = {}
        if verified_predictions:
            avg_error = sum(p['error'] for p in verified_predictions) / len(verified_predictions)
            avg_error_pct = sum(p['error_pct'] for p in verified_predictions) / len(verified_predictions)
            correct_direction = sum(
                1 for p in verified_predictions
                if (p['predicted_price'] > p['latest_price'] and p['actual_price'] > p['latest_price']) or
                   (p['predicted_price'] < p['latest_price'] and p['actual_price'] < p['latest_price'])
            )
            direction_accuracy = (correct_direction / len(verified_predictions)) * 100 if verified_predictions else 0
            accuracy_metrics = {
                'average_error': float(avg_error),
                'average_error_percent': float(avg_error_pct),
                'direction_accuracy_percent': float(direction_accuracy),
                'verified_predictions': len(verified_predictions)
            }
        
        # Muat data untuk prediksi terkini
        data = load_tiingo_data(token_name)
        if data is None or data[0] is None:
            return jsonify({"status": "error", "message": "Tidak ada data harga terkini"})
        df, _ = data
        
        if len(df) == 0:
            return jsonify({"status": "error", "message": "Tidak ada data harga terkini"})
        
        current_price = float(df['close'].iloc[-1])
        current_timestamp = int(df['timestamp'].iloc[-1])
        
        prediction = await asyncio.get_event_loop().run_in_executor(None, cached_prediction, token_name, prediction_horizon)
        if prediction is None:
            return jsonify({"status": "error", "message": "Gagal membuat prediksi"})
        
        change_pct = float(((prediction - current_price) / current_price) * 100)
        
        return jsonify({
            "status": "success",
            "token": token_upper,
            "current_price": float(current_price),
            "current_timestamp": current_timestamp,
            "predicted_price": float(prediction),
            "log_return": float(cache_data.get('log_return', 0.0)),
            "model_type": best_model.upper(),
            "prediction_horizon": f"{prediction_horizon}m",
            "change_percentage": change_pct,
            "data_source": "Tiingo Cache + CoinGecko",
            "accuracy_metrics": accuracy_metrics if accuracy_metrics else None,
            "model_info": {
                "model_type": best_model.upper(),
                "model_path": os.path.join(MODELS_DIR, f'{token_name}_logreturn_{best_model}_model_{prediction_horizon}m.pkl'),
                "look_back_window": TOKEN_INFO[token_name]['look_back'],
                "comparison_data": comparison_data,
                "selection_criteria": "ZPTAE (lebih rendah lebih baik) & Akurasi Arah (lebih tinggi lebih baik)"
            }
        })
    except Exception as e:
        logger.error(f"Error mendapatkan metrik performa: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    MODEL_SELECTION_CACHE.clear()  # Bersihkan cache saat aplikasi dimulai
    logger.info("=" * 50)
    logger.info("PRICE PREDICTION API")
    logger.info("=" * 50)
    
    logger.info("Konfigurasi model:")
    for token, info in TOKEN_INFO.items():
        logger.info(f"  {info['full_name']} ({token}):")
        logger.info(f"    Tipe model: {info['model_type']}")
        logger.info(f"    Horizon prediksi: {info['prediction_horizon']}m")
        logger.info(f"    Jendela look back: {info['look_back']} titik data")
    
    if not os.path.exists(MODELS_DIR):
        logger.warning(f"Direktori model tidak ditemukan: {MODELS_DIR}")
    else:
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl')]
        comparison_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.json')]
        logger.info(f"Ditemukan {len(model_files)} file model dan {len(comparison_files)} file perbandingan")
        
        for token in TOKEN_INFO:
            prediction_horizon = TOKEN_INFO[token]['prediction_horizon']
            best_model = determine_best_model(token, prediction_horizon)
            logger.info(f"Model terbaik untuk {TOKEN_INFO[token]['full_name']} ({prediction_horizon}m): {best_model.upper()}")
    
    os.makedirs(TIINGO_DATA_DIR, exist_ok=True)
    logger.info(f"Direktori cache dibuat: Tiingo={TIINGO_DATA_DIR}")
    
    logger.info(f"Memulai server API pada port {API_PORT}")
    app.run(host='0.0.0.0', port=API_PORT)
