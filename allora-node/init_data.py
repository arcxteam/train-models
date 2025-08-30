import os
import logging
import sys
import pandas as pd
import json
import traceback
from datetime import datetime, timedelta, timezone
from utils import fetch_historical_data
from config import TIINGO_DATA_DIR
import fcntl

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('init_data.log')
    ]
)
logger = logging.getLogger(__name__)

def init_price_token(token_name):
    """
    Inisialisasi data harga token dari Tiingo API menggunakan fetch_historical_data
    Args:
        token_name (str): Nama token (misalnya 'btcusd')
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        logger.info(f"Initializing data for {token_name} from Tiingo API")
        
        # Pastikan direktori cache ada
        os.makedirs(TIINGO_DATA_DIR, exist_ok=True)
        
        # Path file data per token
        json_path = os.path.join(TIINGO_DATA_DIR, f"tiingo_data_5min_{token_name.lower()}.json")
        
        # Cek apakah file sudah ada dan cukup baru
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError as e:
                    logger.error(f"Corrupted JSON in {json_path}: {e}. Attempting to re-fetch data.")
                    os.remove(json_path)
                else:
                    df_temp = pd.DataFrame(data['priceData'])
                    latest_date = pd.to_datetime(df_temp['date']).max() if not df_temp.empty else pd.Timestamp.min
                    current_time = datetime.now(timezone.utc)
                    if not df_temp.empty and (current_time - latest_date).total_seconds() < 24 * 3600:  # Kurang dari 1 hari
                        logger.info(f"Dataset already exists and is recent enough (latest date: {latest_date}). Skipping initialization for {token_name}.")
                        return True
        
        # Hapus cache lama jika ada
        if os.path.exists(json_path):
            os.remove(json_path)
            logger.info(f"Removing old cache: {json_path}")
        
        # Inisialisasi data menggunakan fetch_historical_data dari utils
        df = fetch_historical_data(ticker=token_name, resample_freq="5min")
        if df.empty:
            logger.error(f"No data returned from Tiingo for {token_name}")
            return False
        
        # Simpan data ke file dengan lock
        with open(json_path, 'w') as f:
            fcntl.flock(f, fcntl.LOCK_EX)  # Kunci file
            json.dump({'priceData': df.to_dict('records')}, f, indent=4)
            fcntl.flock(f, fcntl.LOCK_UN)  # Lepas kunci
        logger.info(f"Saved {len(df)} data points to {json_path} for {token_name}, date range: {df['date'].min()} to {df['date'].max()}")
        
        logger.info(f"Successfully initialized {len(df)} data points from Tiingo for {token_name}")
        return True
    
    except Exception as e:
        logger.error(f"Error initializing data for {token_name}: {e}")
        logger.error(traceback.format_exc())
        return False

def initialize_tokens():
    """
    Menginisialisasi semua token yang ditentukan di environment variable
    """
    try:
        tokens = os.environ.get('TOKENS', 'btcusd').split(',')
        logger.info(f"Tokens: {tokens}")
        if tokens and len(tokens) > 0:
            for token in tokens:
                token_parts = token.split(':')
                if len(token_parts) >= 1:
                    token_name = token_parts[0]
                    logger.info(f"Initializing data for {token_name} token")
                    init_price_token(token_name)
    except Exception as e:
        logger.error(f"Failed to initialize tokens: {e}")
    finally:
        logger.info("Tokens initialization completed")
        sys.exit(0)

if __name__ == "__main__":
    initialize_tokens()
