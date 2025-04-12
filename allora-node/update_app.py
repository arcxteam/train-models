import os
import sys
import sqlite3
import time
import logging
import traceback
import pandas as pd
from datetime import datetime, timedelta

# Import dari app_config dan app_utils
from app_config import (
    DATABASE_PATH, CGC_API_KEY, TIINGO_API_TOKEN, 
    TIINGO_CACHE_DIR, OKX_CACHE_DIR
)
from app_utils import (
    get_latest_network_block, check_create_table,
    get_ohlcv_from_tiingo, get_ohlcv_from_okx
)

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('price_updater.log')
    ]
)
logger = logging.getLogger(__name__)

# Import library tergantung pada keberadaannya
try:
    import retrying
    import requests
    import ccxt
    HAS_DEPENDENCIES = True
except ImportError as e:
    logger.warning(f"Missing dependency: {e}")
    HAS_DEPENDENCIES = False

# Function to fetch data with retry
@retrying.retry(wait_exponential_multiplier=1000, wait_exponential_max=10000, stop_max_attempt_number=5)
def fetch_cg_data(url):
    """
    Mengambil data dari CoinGecko API dengan mekanisme retry
    
    Args:
        url: URL CoinGecko API
        
    Returns:
        dict: Data JSON dari respons API
    """
    headers = {
        "accept": "application/json",
        "x-cg-demo-api-key": CGC_API_KEY
    }
    logger.info(f"Fetching data from CoinGecko: {url}")
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def update_price_from_coingecko(token_name, token_from, token_to='usd'):
    """
    Memperbarui harga token dari CoinGecko API
    
    Args:
        token_name: Nama token (berausd)
        token_from: ID token di CoinGecko (berachain)
        token_to: Currency untuk perbandingan (default: usd)
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        check_create_table()
        url = f'https://api.coingecko.com/api/v3/simple/price?ids={token_from}&vs_currencies={token_to}'
        prices = fetch_cg_data(url)

        if token_from.lower() not in prices:
            logger.warning(f"Invalid token ID: {token_from}")
            return False

        price = prices[token_from.lower()][token_to.lower()]
        block_data = get_latest_network_block()
        latest_block_height = int(block_data['block']['header']['height'])
        token = token_name.lower()

        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO prices (block_height, token, price) VALUES (?, ?, ?)", 
                          (latest_block_height, token, price))
            conn.commit()

        logger.info(f"CoinGecko: Inserted data point block {latest_block_height} : {price} for {token_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to update {token_name} price from CoinGecko: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_price_from_tiingo(token_name):
    """
    Memperbarui data OHLCV token dari Tiingo API
    
    Args:
        token_name: Nama token (berausd)
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    try:
        # Pastikan tabel ada dengan struktur OHLCV
        check_create_table()
        
        # Ambil data dari Tiingo API
        df = get_ohlcv_from_tiingo(token_name, resample_freq='1min', days_back=1)
        
        if df is None or len(df) == 0:
            logger.warning(f"No data returned from Tiingo for {token_name}")
            return False
        
        # Get latest network block
        block_data = get_latest_network_block()
        latest_block_height = int(block_data['block']['header']['height'])
        
        # Ambil data terbaru
        latest_data = df.iloc[-1]
        
        # Ekstrak nilai OHLCV
        price = latest_data['close']  # Close price
        open_price = latest_data['open']
        high_price = latest_data['high']
        low_price = latest_data['low']
        volume = latest_data['volume'] if 'volume' in latest_data else 0.0
        timestamp = int(latest_data['timestamp']) if not pd.isna(latest_data['timestamp']) else int(time.time())
        
        # Simpan ke database dengan data OHLCV lengkap
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO prices 
                (block_height, token, price, open, high, low, volume, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                latest_block_height, 
                token_name.lower(), 
                price, 
                open_price, 
                high_price, 
                low_price, 
                volume,
                timestamp
            ))
            conn.commit()
        
        logger.info(f"Tiingo: Inserted OHLCV data at block {latest_block_height} for {token_name}")
        
        # Untuk beberapa data terbaru, simpan juga block heights sebelumnya (untuk history)
        if len(df) > 1:
            # Ambil beberapa data terbaru (hingga 10) untuk history
            recent_data = df.iloc[-10:] if len(df) >= 10 else df
            
            logger.info(f"Saving {len(recent_data)} historical OHLCV data points for {token_name}")
            
            for idx, row in recent_data.iterrows():
                if idx == len(recent_data) - 1:  # Skip data terbaru (sudah diinsert)
                    continue
                    
                # Hitung block height berdasarkan perbedaan waktu
                time_diff = (latest_data['timestamp'] - row['timestamp']) / 60  # dalam menit
                hist_block_height = int(latest_block_height - (time_diff / 5))  # asumsi 1 block = 5 menit
                
                # Simpan ke database jika block height berbeda
                if hist_block_height != latest_block_height:
                    # Ekstrak nilai OHLCV untuk data historis
                    hist_price = row['close']
                    hist_open = row['open']
                    hist_high = row['high']
                    hist_low = row['low']
                    hist_volume = row['volume'] if 'volume' in row else 0.0
                    hist_timestamp = int(row['timestamp']) if not pd.isna(row['timestamp']) else 0
                    
                    with sqlite3.connect(DATABASE_PATH) as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                            INSERT OR IGNORE INTO prices 
                            (block_height, token, price, open, high, low, volume, timestamp) 
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            hist_block_height, 
                            token_name.lower(), 
                            hist_price, 
                            hist_open, 
                            hist_high, 
                            hist_low, 
                            hist_volume,
                            hist_timestamp
                        ))
                        conn.commit()
                        
                    logger.info(f"Tiingo: Added historical OHLCV data at block {hist_block_height} for {token_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to update {token_name} OHLCV data from Tiingo: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_price_from_okx(token_name, symbol=None):
    """
    Memperbarui data OHLCV token dari OKX API
    
    Args:
        token_name: Nama token (berausd)
        symbol: Simbol trading di OKX (default: None, akan dihasilkan dari token_name)
        
    Returns:
        bool: True jika berhasil, False jika gagal
    """
    if not HAS_DEPENDENCIES:
        logger.error("Cannot use OKX update: missing dependencies")
        return False
        
    try:
        # Pastikan tabel ada dengan struktur OHLCV
        check_create_table()
        
        # Jika symbol tidak ditentukan, buat dari token_name
        if symbol is None:
            # Konversi berausd -> BERA/USDT
            if token_name.lower() == 'berausd':
                symbol = 'BERA/USDT'
            else:
                # Format umum: tokenusd -> TOKEN/USDT
                symbol = f"{token_name.replace('usd', '').upper()}/USDT"
        
        # Ambil data dari OKX API
        df = get_ohlcv_from_okx(symbol, timeframe='1m', limit=100)
        
        if df is None or len(df) == 0:
            logger.warning(f"No data returned from OKX for {symbol}")
            return False
        
        # Get latest network block
        block_data = get_latest_network_block()
        latest_block_height = int(block_data['block']['header']['height'])
        
        # Ambil data terbaru
        latest_data = df.iloc[-1]
        
        # Ekstrak nilai OHLCV
        price = latest_data['close']  # Close price = price
        open_price = latest_data['open']
        high_price = latest_data['high']
        low_price = latest_data['low']
        volume = latest_data['volume'] if 'volume' in latest_data else 0.0
        timestamp = int(latest_data['timestamp']) if not pd.isna(latest_data['timestamp']) else int(time.time())
        
        # Simpan ke database dengan data OHLCV lengkap
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO prices 
                (block_height, token, price, open, high, low, volume, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                latest_block_height, 
                token_name.lower(), 
                price, 
                open_price, 
                high_price, 
                low_price, 
                volume,
                timestamp
            ))
            conn.commit()
        
        logger.info(f"OKX: Inserted OHLCV data at block {latest_block_height} for {token_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to update {token_name} OHLCV data from OKX: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def update_price(token_name, token_from, token_to='usd'):
    """
    Memperbarui harga token dengan strategi multi-sumber
    1. Coba Tiingo API
    2. Jika gagal, coba OKX API
    3. Jika gagal, fallback ke CoinGecko API
    
    Args:
        token_name: Nama token (berausd)
        token_from: ID token di CoinGecko (berachain)
        token_to: Currency untuk perbandingan (default: usd)
    """
    logger.info(f"Updating price for {token_name} using multi-source strategy")
    
    # 1. Coba update dari Tiingo API (prioritas utama)
    tiingo_success = update_price_from_tiingo(token_name)
    
    if tiingo_success:
        logger.info(f"Successfully updated {token_name} price from Tiingo API")
        return
    
    # 2. Jika Tiingo gagal, coba dari OKX
    logger.info(f"Tiingo update failed, trying OKX for {token_name}")
    okx_success = update_price_from_okx(token_name)
    
    if okx_success:
        logger.info(f"Successfully updated {token_name} price from OKX API")
        return
    
    # 3. Jika OKX gagal, fallback ke CoinGecko
    logger.info(f"OKX update failed, falling back to CoinGecko for {token_name}")
    cg_success = update_price_from_coingecko(token_name, token_from, token_to)
    
    if cg_success:
        logger.info(f"Successfully updated {token_name} price from CoinGecko API")
        return
    
    # Jika semua gagal
    logger.error(f"Failed to update {token_name} price from all sources")

def clean_old_cache():
    """
    Membersihkan file cache yang sudah lama untuk menghemat ruang disk
    """
    try:
        # Bersihkan cache Tiingo yang lebih dari 1 hari
        if os.path.exists(TIINGO_CACHE_DIR):
            for filename in os.listdir(TIINGO_CACHE_DIR):
                file_path = os.path.join(TIINGO_CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if time.time() - file_time > 172800:  # 48 jam
                        os.remove(file_path)
                        logger.info(f"Cleaned old Tiingo cache file: {filename}")
        
        # Bersihkan cache OKX yang lebih dari 1 hari
        if os.path.exists(OKX_CACHE_DIR):
            for filename in os.listdir(OKX_CACHE_DIR):
                file_path = os.path.join(OKX_CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    file_time = os.path.getmtime(file_path)
                    if time.time() - file_time > 172800:  # 48 jam
                        os.remove(file_path)
                        logger.info(f"Cleaned old OKX cache file: {filename}")
    
    except Exception as e:
        logger.error(f"Error cleaning cache: {e}")

def main():
    """
    Fungsi utama untuk menjalankan script update harga
    """
    try:
        logger.info("Starting price update process")
        
        # Saat startup, bersihkan cache lama
        clean_old_cache()
        
        # Process tokens from environment variable
        tokens = os.environ.get('TOKENS', '').split(',')
        
        if not tokens or tokens[0] == '':
            logger.error("No tokens specified in TOKENS environment variable")
            sys.exit(1)
            
        # Untuk setiap token, update harga
        for token_spec in tokens:
            if ':' not in token_spec:
                logger.warning(f"Invalid token specification: {token_spec}, skipping")
                continue
                
            token_parts = token_spec.split(':')
            if len(token_parts) != 2:
                logger.warning(f"Invalid token format: {token_spec}, expected format TOKEN:CGID")
                continue
                
            token_name = token_parts[0]
            if 'usd' not in token_name.lower():
                token_name = f"{token_name}USD"
            token_cg_id = token_parts[1]
            
            logger.info(f"Processing {token_name} (CoinGecko ID: {token_cg_id})")
            update_price(token_name, token_cg_id, 'usd')
            
            # Delay kecil antar token jika ada banyak
            if len(tokens) > 1:
                time.sleep(1)
        
        logger.info("Price update process completed successfully")
        
    except KeyError as e:
        logger.error(f"Environment variable {str(e)} not found.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        sys.exit(0)

if __name__ == "__main__":
    main()
