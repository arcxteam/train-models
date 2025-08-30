import os
import logging
import time
import pandas as pd
import json
import traceback
from datetime import datetime, timedelta
from config import TIINGO_DATA_DIR
from utils import update_recent_data
import fcntl

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('price_updater.log')]
)
logger = logging.getLogger(__name__)

def update_price_from_tiingo(token_name):
    """
    Memperbarui data token dari Tiingo API setiap 5 menit dengan buffer 7 hari
    
    Args:
        token_name (str): Nama token (misalnya 'btcusd')
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        # Ambil data terbaru (5 menit terakhir)
        df = update_recent_data(token_name, "5min")  # Menggunakan update_recent_data dari utils
        if df is None or len(df) == 0:
            logger.warning(f"No data returned from Tiingo for {token_name}")
            return False

        json_path = os.path.join(TIINGO_DATA_DIR, f"tiingo_data_5min_{token_name.lower()}.json")
        
        # Cek apakah file cache ada
        if not os.path.exists(json_path):
            logger.error(f"Cache file not found: {json_path}")
            return False
        
        # Cek apakah file sedang diinisialisasi
        try:
            with open(json_path, 'a') as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)  # Non-blocking lock
                fcntl.flock(lock_file, fcntl.LOCK_UN)
        except IOError:
            logger.info(f"File {json_path} is being initialized, skipping update")
            return True

        # Muat data existing
        with open(json_path, 'r') as f:
            existing_data = json.load(f)
        df_existing = pd.DataFrame(existing_data['priceData'])
        latest_existing_date = pd.to_datetime(df_existing['date']).max()
        latest_new_date = pd.to_datetime(df['date']).iloc[0] if not df.empty else None

        if latest_new_date and latest_existing_date == latest_new_date:
            logger.info(f"No new data for {token_name}, skipping update")
        else:
            df_new = pd.concat([df_existing, df]).sort_values("date").drop_duplicates(subset=['date'], keep='last')
            # Buffer 7 hari
            max_date = pd.to_datetime(df_new['date'].max())
            cutoff_date = max_date - timedelta(days=7)
            cutoff_date_str = cutoff_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')
            df_new = df_new[df_new['date'] >= cutoff_date_str]
            logger.info(f"Added new OHLCV records for {token_name}, total records: {len(df_new)}")

            existing_data['priceData'] = df_new.to_dict('records')
            with open(json_path, 'w') as f:
                json.dump(existing_data, f, indent=4)
            logger.info(f"Updated dataset at {json_path}, total records: {len(df_new)}")

        return True
    except Exception as e:
        logger.error(f"Failed to update {token_name} data from Tiingo: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    """
    Fungsi utama untuk menjalankan pembaruan harga setiap 5 menit
    """
    last_added_timestamp = None
    while True:
        try:
            logger.info("Starting price update process at %s", datetime.utcnow())
            tokens = os.environ.get('TOKENS', 'btcusd').split(',')
            if not tokens or tokens[0] == '':
                logger.error("No tokens specified in TOKENS environment variable")
                time.sleep(250)
                continue
            for token_spec in tokens:
                token_parts = token_spec.split(':')
                if len(token_parts) >= 1:
                    token_name = token_parts[0]
                    logger.info(f"Processing {token_name}")
                    update_price_from_tiingo(token_name)
                if len(tokens) > 5:
                    time.sleep(5)
            logger.info("Price update process completed successfully")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
        logger.info("Waiting 4~5 minutes before next update dataset point")
        time.sleep(250)

if __name__ == "__main__":
    main()
