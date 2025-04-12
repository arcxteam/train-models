import os
import requests
import json
import numpy as np
import pandas as pd
import sqlite3
import time
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from zipfile import ZipFile

# Import dari app_config
from app_config import (
    DATABASE_PATH, BLOCK_TIME_SECONDS, DATA_BASE_PATH, 
    ALLORA_VALIDATOR_API_URL, URL_QUERY_LATEST_BLOCK,
    TIINGO_API_TOKEN, TIINGO_CACHE_DIR, TIINGO_CACHE_TTL, OKX_CACHE_DIR
)

# Inisialisasi logger
logger = logging.getLogger(__name__)

# Cek dan buat direktori cache jika belum ada
os.makedirs(TIINGO_CACHE_DIR, exist_ok=True)
os.makedirs(OKX_CACHE_DIR, exist_ok=True)

# Variabel untuk mencatat waktu permintaan API dan rate limiting
_last_tiingo_request_time = 0
_tiingo_request_count_hourly = 0
_tiingo_request_count_daily = 0
_tiingo_hourly_reset = time.time()
_tiingo_daily_reset = time.time()

# Inisialisasi OKX client
okx_client = None
try:
    import ccxt
    okx_client = ccxt.okx()
    logger.info("OKX client initialized successfully")
except ImportError:
    logger.warning("ccxt library not available, OKX data source will be disabled")
except Exception as e:
    logger.error(f"Failed to initialize OKX client: {e}")

# Function to check and create the table if not exists
def check_create_table():
    """
    Memeriksa dan membuat tabel prices dengan data OHLCV lengkap jika belum ada
    """
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Cek apakah tabel prices sudah ada
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='prices'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                # Cek apakah perlu migrasi dari format lama
                cursor.execute("PRAGMA table_info(prices)")
                columns = [col[1] for col in cursor.fetchall()]
                
                # Jika tabel lama hanya memiliki struktur awal
                if set(columns) == set(['block_height', 'token', 'price']):
                    logger.info("Detected old table format, migrating to OHLCV format...")
                    
                    # Cadangkan data lama
                    cursor.execute("CREATE TABLE prices_backup AS SELECT * FROM prices")
                    logger.info("Backed up old data to prices_backup")
                    
                    # Drop tabel lama
                    cursor.execute("DROP TABLE prices")
                    
                    # Buat tabel baru dengan struktur OHLCV
                    cursor.execute('''
                        CREATE TABLE prices (
                            block_height INTEGER,
                            token TEXT,
                            price REAL,
                            open REAL,
                            high REAL,
                            low REAL,
                            volume REAL,
                            timestamp INTEGER,
                            PRIMARY KEY (block_height, token)
                        )
                    ''')
                    
                    # Masukkan data lama, dengan nilai OHLCV dummy
                    cursor.execute('''
                        INSERT INTO prices (block_height, token, price, open, high, low, volume, timestamp)
                        SELECT block_height, token, price, price, price, price, 0, 
                               (strftime('%s','now') - ((block_height % 10000) * 300)) 
                        FROM prices_backup
                    ''')
                    
                    logger.info("Migration completed: Table upgraded to OHLCV format")
                    conn.commit()
            else:
                # Buat tabel baru jika belum ada
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS prices (
                        block_height INTEGER,
                        token TEXT,
                        price REAL,
                        open REAL,
                        high REAL,
                        low REAL,
                        volume REAL,
                        timestamp INTEGER,
                        PRIMARY KEY (block_height, token)
                    )
                ''')
                logger.info("Created new prices table with OHLCV structure")
                conn.commit()
                
    except sqlite3.Error as e:
        logger.error(f"An error occurred while creating/updating the table: {str(e)}")

def download_binance_data(symbol, interval, year, month, download_path):
    """
    Download data Binance untuk simbol dan interval tertentu
    
    Args:
        symbol (str): Simbol trading Binance
        interval (str): Interval waktu ('1m', '1h', etc.)
        year (int): Tahun data
        month (int): Bulan data
        download_path (str): Path untuk menyimpan file yang didownload
    """
    base_url = f"https://data.binance.vision/data/futures/um/daily/klines"
    with ThreadPoolExecutor() as executor:
        for day in range(1, 32):  # Asumsi hari 1-31
            url = f"{base_url}/{symbol}/{interval}/{symbol}-{interval}-{year}-{month:02d}-{day:02d}.zip"
            executor.submit(download_url, url, download_path)

def download_url(url, download_path):
    """
    Download file dari URL ke path tertentu
    
    Args:
        url (str): URL file yang akan didownload
        download_path (str): Path tujuan
    """
    target_file_path = os.path.join(download_path, os.path.basename(url)) 
    if os.path.exists(target_file_path):
        logger.info(f"File already exists: {url}")
        return
    
    response = requests.get(url)
    if response.status_code == 404:
        logger.info(f"File does not exist: {url}")
    else:
        os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
        with open(target_file_path, "wb") as f:
            f.write(response.content)
        logger.info(f"Downloaded: {url} to {target_file_path}")

def extract_and_process_binance_data(token_name, download_path, start_date_epoch, end_date_epoch, latest_block_height):
    """
    Ekstrak dan proses data Binance ke database
    
    Args:
        token_name (str): Nama token
        download_path (str): Path file yang didownload
        start_date_epoch (int): Timestamp untuk batas awal data
        end_date_epoch (int): Timestamp untuk batas akhir data
        latest_block_height (int): Block height terbaru
    """
    files = sorted([x for x in os.listdir(download_path) if x.endswith('.zip')])

    if len(files) == 0:
        logger.warning(f"No data files found for {token_name}")
        return

    with sqlite3.connect(DATABASE_PATH) as conn:
        cursor = conn.cursor()

        for file in files:
            zip_file_path = os.path.join(download_path, file)

            try:
                with ZipFile(zip_file_path) as myzip:
                    with myzip.open(myzip.filelist[0]) as f:
                        df = pd.read_csv(f, header=None)
                        df.columns = [
                            "open_time", "open", "high", "low", "close",
                            "volume", "close_time", "quote_volume", 
                            "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"
                        ]
                        
                        df['close_time'] = pd.to_numeric(df['close_time'], errors='coerce')
                        df.dropna(subset=['close_time'], inplace=True)
                        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')

                        for _, row in df.iterrows():
                            price_timestamp = row['close_time'].timestamp()
                            if price_timestamp < start_date_epoch or price_timestamp > end_date_epoch:
                                continue

                            blocks_diff = (end_date_epoch - price_timestamp) / BLOCK_TIME_SECONDS
                            block_height = int(latest_block_height - blocks_diff)

                            if block_height < 1:
                                continue

                            price = row['close']
                            cursor.execute("INSERT OR REPLACE INTO prices (block_height, token, price) VALUES (?, ?, ?)", 
                                           (block_height, token_name.lower(), price))
                            logger.info(f"{token_name} - {price_timestamp} - Inserted data point - block {block_height} : {price}")

            except Exception as e:
                logger.error(f"Error reading {zip_file_path}: {str(e)}")
                continue

        conn.commit()

def get_latest_network_block():
    """
    Mendapatkan block height terbaru dari network
    
    Returns:
        dict: Data block dengan block height
    """
    try:
        url = f"{ALLORA_VALIDATOR_API_URL}{URL_QUERY_LATEST_BLOCK}"
        logger.info(f"Latest network block URL: {url}")
        response = requests.get(url)
        response.raise_for_status()

         # Handle case where the response might be a list or dictionary
        if isinstance(response.json(), list):
            block_data = response.json()[0]  # Assume it's a list, get the first element
        else:
            block_data = response.json()  # Assume it's already a dictionary

        try:
            latest_block_height = int(block_data['block']['header']['height'])
            logger.info(f"Latest block height: {latest_block_height}")
        except KeyError:
            logger.error("Error: Missing expected keys in block data.")
            latest_block_height = 0

        return {'block': {'header': {'height': latest_block_height}}}
    except Exception as e:
        logger.error(f'Failed to get block height: {str(e)}')
        return {}

def init_price_token(symbol, token_name, token_to):
    """
    Inisialisasi data harga token dari Binance
    
    Args:
        symbol (str): Simbol trading (seperti 'BERA')
        token_name (str): Nama token (seperti 'berausd')
        token_to (str): Pasangan quote currency (seperti 'USD')
    """
    try:
        check_create_table()

        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM prices WHERE token=?", (token_name.lower(),))
            count = cursor.fetchone()[0]

        if count > 15000:
            logger.info(f'Data already exists for {token_name} token, {count} entries')
            return
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=31)

        block_data = get_latest_network_block()
        latest_block_height = int(block_data['block']['header']['height'])

        start_date_epoch = int(start_date.timestamp())
        end_date_epoch = int(end_date.timestamp())

        symbol = f"{symbol.upper()}{token_to.upper()}T"
        interval = "1m"  # 1-minute interval data
        binance_data_path = os.path.join(DATA_BASE_PATH, "binance/futures-klines")
        download_path = os.path.join(binance_data_path, symbol.lower())
        download_binance_data(symbol, interval, end_date.year, end_date.month, download_path)
        extract_and_process_binance_data(token_name, download_path, start_date_epoch, end_date_epoch, latest_block_height)

        logger.info(f'Data initialized successfully for {token_name} token')
    except Exception as e:
        logger.error(f'Failed to initialize data for {token_name} token: {str(e)}')
        raise e

def _check_tiingo_rate_limits():
    """
    Memeriksa batas rate Tiingo API dan melakukan reset jika perlu
    
    Returns:
        bool: True jika permintaan masih di bawah batas, False jika batas tercapai
    """
    global _tiingo_request_count_hourly, _tiingo_request_count_daily
    global _tiingo_hourly_reset, _tiingo_daily_reset
    
    current_time = time.time()
    
    # Reset penghitung jam setiap jam
    if current_time - _tiingo_hourly_reset > 3600:  # 1 jam dalam detik
        _tiingo_request_count_hourly = 0
        _tiingo_hourly_reset = current_time
        logger.info("Tiingo hourly request counter reset")
    
    # Reset penghitung harian setiap 24 jam
    if current_time - _tiingo_daily_reset > 86400:  # 24 jam dalam detik
        _tiingo_request_count_daily = 0
        _tiingo_daily_reset = current_time
        logger.info("Tiingo daily request counter reset")
    
    # Periksa batas
    if _tiingo_request_count_hourly >= 45:  # Tetapkan batas 45 (dari 50) untuk jaga-jaga
        logger.warning("Approaching Tiingo hourly request limit (45/50). Using cached data.")
        return False
    
    if _tiingo_request_count_daily >= 900:  # Tetapkan batas 900 (dari 1000) untuk jaga-jaga
        logger.warning("Approaching Tiingo daily request limit (900/1000). Using cached data.")
        return False
    
    return True

def _increment_tiingo_counters():
    """Menambah penghitung permintaan Tiingo"""
    global _tiingo_request_count_hourly, _tiingo_request_count_daily
    _tiingo_request_count_hourly += 1
    _tiingo_request_count_daily += 1

def _get_tiingo_cache_path(ticker, resample_freq):
    """
    Mendapatkan path cache untuk data Tiingo
    
    Args:
        ticker (str): Ticker simbol
        resample_freq (str): Frekuensi data
        
    Returns:
        str: Path file cache
    """
    return os.path.join(TIINGO_CACHE_DIR, f"{ticker}_{resample_freq}_cache.json")

def _is_cache_valid(cache_path):
    """
    Memeriksa apakah file cache masih valid berdasarkan TTL
    
    Args:
        cache_path (str): Path file cache
        
    Returns:
        bool: True jika cache valid, False jika tidak
    """
    if not os.path.exists(cache_path):
        return False
    
    file_mtime = os.path.getmtime(cache_path)
    current_time = time.time()
    
    return (current_time - file_mtime) < TIINGO_CACHE_TTL

def get_ohlcv_from_tiingo(ticker, resample_freq='1min', days_back=1):
    """
    Mengambil data OHLCV dari Tiingo API untuk kripto
    
    Args:
        ticker (str): Simbol ticker (misalnya 'berausd')
        resample_freq (str): Frekuensi data ('1min', '5min', '1hour', etc.)
        days_back (int): Jumlah hari mundur untuk data
        
    Returns:
        pandas.DataFrame: Data OHLCV atau None jika gagal
    """
    try:
        # Pastikan kita menggunakan format ticker yang benar untuk Tiingo
        base_ticker = ticker.lower()
        # Hapus 'usd' jika ada, terlepas dari huruf besar/kecil
        if 'usd' in base_ticker:
            base_ticker = base_ticker.replace('usd', '')
        ticker_formatted = f"{base_ticker}usd"  # Format yang diharapkan Tiingo
        
        logger.info(f"Formatting ticker from {ticker} to {ticker_formatted} for Tiingo API")
        
        # Cek file cache
        cache_path = _get_tiingo_cache_path(ticker_formatted, resample_freq)
        
        # Jika cache valid dan batas rate hampir tercapai, gunakan cache
        if _is_cache_valid(cache_path) and not _check_tiingo_rate_limits():
            try:
                logger.info(f"Using cached Tiingo data from {cache_path}")
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Konversi cache ke DataFrame
                df = pd.DataFrame(cached_data)
                return df
            except Exception as e:
                logger.error(f"Error reading Tiingo cache: {e}")
                # Lanjutkan dengan permintaan API jika cache gagal
        
        # Cek rate limit
        if not _check_tiingo_rate_limits():
            logger.warning("Tiingo rate limit approached. Skipping request.")
            return None
        
        # Hitung tanggal mulai (days_back hari yang lalu)
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        # Siapkan URL dan headers
        url = f"https://api.tiingo.com/tiingo/crypto/prices"
        params = {
            'tickers': ticker_formatted,
            'startDate': start_date,
            'resampleFreq': resample_freq,
            'token': TIINGO_API_TOKEN
        }
        headers = {
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Fetching data from Tiingo for {ticker_formatted} with {resample_freq} interval")
        
        # Buat request ke Tiingo API
        global _last_tiingo_request_time
        
        # Menambahkan delay antara permintaan untuk menghindari rate limiting
        current_time = time.time()
        if current_time - _last_tiingo_request_time < 3.0:  # 3 detik delay
            time.sleep(3.0 - (current_time - _last_tiingo_request_time))
        
        response = requests.get(url, headers=headers, params=params)
        _last_tiingo_request_time = time.time()
        _increment_tiingo_counters()
        
        # Log response untuk debugging
        logger.info(f"Tiingo API response status: {response.status_code}")
        if response.status_code != 200:
            logger.warning(f"Tiingo API error: {response.text}")
        
        response.raise_for_status()  # Raise exception jika terjadi error
        
        # Parse respons JSON
        data = response.json()
        
        if not data or len(data) == 0:
            logger.warning(f"Empty response from Tiingo API for {ticker_formatted}")
            return None
            
        if 'priceData' not in data[0] or len(data[0]['priceData']) == 0:
            logger.warning(f"No price data in Tiingo response for {ticker_formatted}")
            return None
        
        # Extract data OHLCV
        price_data = data[0]['priceData']
        
        # Simpan ke cache
        try:
            with open(cache_path, 'w') as f:
                json.dump(price_data, f)
            logger.info(f"Cached Tiingo data to {cache_path}")
        except Exception as e:
            logger.error(f"Error caching Tiingo data: {e}")
        
        # Konversi ke DataFrame
        df = pd.DataFrame(price_data)
        
        # Standarisasi kolom untuk sesuai dengan format yang diperlukan model
        df['price'] = df['close']  # Tambahkan kolom price yang sama dengan close
        df['timestamp'] = pd.to_datetime(df['date']).astype(int) / 10**9  # Konversi ke UNIX timestamp dalam detik
        
        logger.info(f"Successfully fetched {len(df)} records from Tiingo")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data from Tiingo: {e}")
        logger.error(traceback.format_exc())
        return None

def get_ohlcv_from_okx(symbol, timeframe='1m', limit=100):
    """
    Mengambil data OHLCV terbaru dari OKX API
    
    Args:
        symbol (str): Simbol trading, mis. 'BERA/USDT'
        timeframe (str): Interval waktu ('1m', '5m', '15m', '1h', etc.)
        limit (int): Jumlah data yang diambil
        
    Returns:
        pandas.DataFrame: Data OHLCV atau None jika gagal
    """
    if okx_client is None:
        logger.error("OKX client not initialized")
        return None
    
    try:
        # Cek cache OKX
        cache_filename = f"{symbol.replace('/', '_')}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
        cache_path = os.path.join(OKX_CACHE_DIR, cache_filename)
        
        # Periksa apakah cache masih valid (kurang dari 5 menit)
        if os.path.exists(cache_path):
            file_mtime = os.path.getmtime(cache_path)
            if time.time() - file_mtime < 300:  # 5 menit dalam detik
                try:
                    logger.info(f"Using cached OKX data from {cache_path}")
                    df = pd.read_csv(cache_path)
                    return df
                except Exception as e:
                    logger.error(f"Error reading OKX cache: {e}")
                    # Lanjutkan dengan permintaan API jika cache gagal
        
        # Ambil data OHLCV dari OKX
        logger.info(f"Fetching {limit} {timeframe} OHLCV data for {symbol} from OKX")
        ohlcv = okx_client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        
        if not ohlcv or len(ohlcv) == 0:
            logger.warning(f"No OHLCV data returned for {symbol}")
            return None
        
        # Konversi ke DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Konversi timestamp dan tambahkan kolom yang sesuai dengan format data training
        df['timestamp'] = df['timestamp'] / 1000  # Konversi dari ms ke detik
        df['price'] = df['close']  # Nama kolom yang sama dengan data training
        
        # Simpan ke cache
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            df.to_csv(cache_path, index=False)
            logger.info(f"Cached OKX data to {cache_path}")
        except Exception as e:
            logger.error(f"Error caching OKX data: {e}")
        
        logger.info(f"Successfully fetched {len(df)} OHLCV records for {symbol}")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching OHLCV data from OKX: {e}")
        logger.error(traceback.format_exc())
        return None

def load_data_from_db(token_name, limit=None):
    """
    Load OHLCV data from SQLite database
    
    Args:
        token_name (str): Token name (lowercase, e.g., 'berausd')
        limit (int): Number of data points to return
        
    Returns:
        tuple: (prices, block_heights, timestamps, ohlcv_data) or (prices, block_heights, timestamps) for compatibility
    """
    try:
        with sqlite3.connect(DATABASE_PATH) as conn:
            cursor = conn.cursor()
            
            # Check if OHLCV columns exist
            cursor.execute("PRAGMA table_info(prices)")
            columns = [col[1] for col in cursor.fetchall()]
            has_ohlcv = all(col in columns for col in ['open', 'high', 'low', 'volume'])
            
            if has_ohlcv:
                # Use OHLCV format
                query = """
                    SELECT price, block_height, timestamp, open, high, low, volume 
                    FROM prices 
                    WHERE token=?
                    ORDER BY block_height DESC
                """
            else:
                # Use old format
                query = """
                    SELECT price, block_height, timestamp FROM prices 
                    WHERE token=? AND timestamp IS NOT NULL
                    ORDER BY block_height DESC
                """
                if 'timestamp' not in columns:
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
            
            if has_ohlcv and data.shape[1] > 6:
                timestamps = data[:, 2]
                
                # Extract OHLCV data
                df = pd.DataFrame({
                    'price': data[:, 0],
                    'timestamp': data[:, 2],
                    'open': data[:, 3],
                    'high': data[:, 4],
                    'low': data[:, 5],
                    'volume': data[:, 6]
                })
                
                logger.info(f"Loaded {len(prices)} OHLCV data points from database for {token_name}")
                return prices, block_heights, timestamps, df
            
            elif data.shape[1] > 2:
                timestamps = data[:, 2]
            else:
                timestamps = np.array([None] * len(prices))
                
            logger.info(f"Loaded {len(prices)} data points from database for {token_name}")
            # For compatibility with old code
            return prices, block_heights, timestamps
        else:
            logger.warning(f"No data found in database for {token_name}")
            return None, None, None
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        logger.error(traceback.format_exc())
        return None, None, None
        
def prepare_features_for_prediction(df, scaler):
       """
       Menyiapkan fitur untuk prediksi model log-return dengan 17 fitur standar
       
       Args:
           df (DataFrame): Data OHLCV yang sudah diurutkan berdasarkan timestamp
           scaler (MinMaxScaler): Scaler yang sudah di-fit pada data training
           
       Returns:
           numpy.array: Array fitur yang telah di-scale (17 fitur)
       """
       try:
           logger.info(f"Menyiapkan fitur prediksi dari {len(df)} titik data")
           
           # Pastikan semua kolom yang diperlukan ada
           required_cols = ['price', 'open', 'high', 'low']
           missing_cols = [col for col in required_cols if col not in df.columns]
           if missing_cols:
               logger.warning(f"Kolom yang diperlukan tidak ada: {missing_cols}")
               if 'price' in df.columns:
                   for col in missing_cols:
                       if col != 'price':
                           df[col] = df['price']
               else:
                   logger.error("Tidak dapat menyiapkan fitur: kolom 'price' tidak ada")
                   return None
           
           # Persiapkan fitur teknikal (sama seperti di prepare_data_for_log_return)
           # 1. Price range
           df['price_range'] = df['high'] - df['low']
           
           # 2. Body size
           df['body_size'] = abs(df['price'] - df['open'])
           
           # 3. Price position
           df['price_position'] = (df['price'] - df['low']) / df['price_range'].replace(0, 1)
           
           # 4. Normalized volume
           if 'volume' in df.columns:
               df['norm_volume'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
           else:
               df['norm_volume'] = 0.0
           
           # 5. Simple Moving Averages
           df['sma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
           df['sma_10'] = df['price'].rolling(window=10, min_periods=1).mean()
           df['sma_20'] = df['price'].rolling(window=20, min_periods=1).mean()
           
           # 6. Distance from SMAs
           df['dist_sma_5'] = (df['price'] - df['sma_5']) / df['price']
           df['dist_sma_10'] = (df['price'] - df['sma_10']) / df['price']
           df['dist_sma_20'] = (df['price'] - df['sma_20']) / df['price']
           
           # 7. Momentum
           df['momentum_5'] = df['price'].pct_change(periods=5)
           df['momentum_10'] = df['price'].pct_change(periods=10)
           df['momentum_20'] = df['price'].pct_change(periods=20)
           
           # Isi NaN values
           df.fillna(0, inplace=True)
           
           # Daftar fitur yang sama seperti saat training
           feature_columns = [
               'price', 'open', 'high', 'low', 'price_range',
               'body_size', 'price_position', 'norm_volume',
               'sma_5', 'dist_sma_5', 'sma_10', 'dist_sma_10',
               'sma_20', 'dist_sma_20', 'momentum_5',
               'momentum_10', 'momentum_20'
           ]
           
           # Ambil baris terakhir (data terbaru)
           latest_features = df.iloc[-1][feature_columns].values.reshape(1, -1)
           
           # Terapkan scaling yang sama dengan training
           scaled_features = scaler.transform(latest_features)
           
           logger.info(f"Berhasil menyiapkan fitur prediksi dengan shape: {scaled_features.shape}")
           return scaled_features
       
       except Exception as e:
           logger.error(f"Error menyiapkan fitur prediksi: {str(e)}")
           logger.error(traceback.format_exc())
           return None

def prepare_data_for_lstm(df, look_back, prediction_horizon, feature_cols=None):
    """
    Menyiapkan data untuk model LSTM
    
    Args:
        df (DataFrame): Data OHLCV dengan minimal kolom 'price' dan 'timestamp'
        look_back (int): Jumlah time steps untuk melihat ke belakang
        prediction_horizon (int): Jarak prediksi (dalam time steps)
        feature_cols (list): Daftar kolom fitur untuk digunakan, default None
        
    Returns:
        tuple: (X, y, scaler, timestamps) 
               X = fitur untuk model, 
               y = target, 
               scaler = scaler yang digunakan, 
               timestamps = timestamps untuk data
    """
    try:
        logger.info(f"Preparing data for LSTM with look_back={look_back}, prediction_horizon={prediction_horizon}")
        
        # Jika tidak ada kolom fitur yang ditentukan, gunakan harga saja
        if feature_cols is None:
            feature_cols = ['price']
        
        # Pastikan df diurutkan berdasarkan timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Ambil kolom fitur dan nilai target
        data = df[feature_cols].values
        
        # Buat array untuk X dan y
        X = []
        y = []
        timestamps = []
        
        # Untuk setiap time step yang cukup untuk look back dan prediction horizon
        for i in range(len(data) - look_back - prediction_horizon):
            # Ambil look_back data points
            X.append(data[i:(i + look_back)])
            # Target adalah prediction_horizon time steps ke depan
            y.append(data[i + look_back + prediction_horizon - 1][0])  # Asumsi target adalah kolom pertama (price)
            # Simpan timestamp untuk data ini
            timestamps.append(df['timestamp'].iloc[i + look_back])
        
        # Konversi ke numpy arrays
        X = np.array(X)
        y = np.array(y)
        timestamps = np.array(timestamps)
        
        logger.info(f"Prepared LSTM data: X shape = {X.shape}, y shape = {y.shape}")
        
        # Untuk LSTM, reshape X ke [samples, time_steps, features]
        if len(X.shape) == 2:
            X = X.reshape((X.shape[0], X.shape[1], 1))
        
        return X, y, None, timestamps
        
    except Exception as e:
        logger.error(f"Error preparing data for LSTM: {e}")
        logger.error(traceback.format_exc())
        return None, None, None, None

def prepare_data_for_log_return(df, look_back, prediction_horizon):
    """
    Menyiapkan data untuk model log-return (XGBoost/Random Forest)
    dengan 17 fitur teknikal standar
    
    Args:
        df (DataFrame): Data OHLCV dengan minimal kolom 'price', 'open', 'high', 'low', 'volume', 'timestamp'
        look_back (int): Jumlah time steps untuk melihat ke belakang
        prediction_horizon (int): Jarak prediksi (dalam time steps)
        
    Returns:
        tuple: (X, y, scaler, timestamps)
               X = fitur untuk model (17 fitur teknikal), 
               y = target (future log return), 
               scaler = MinMaxScaler yang digunakan,
               timestamps = timestamps untuk data
    """
    try:
        logger.info(f"Preparing data for log-return model with look_back={look_back}, prediction_horizon={prediction_horizon}")
        
        # Pastikan df diurutkan berdasarkan timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Daftar fitur yang akan digunakan
        feature_columns = [
            'price', 'open', 'high', 'low', 'price_range',
            'body_size', 'price_position', 'norm_volume',
            'sma_5', 'dist_sma_5', 'sma_10', 'dist_sma_10',
            'sma_20', 'dist_sma_20', 'momentum_5',
            'momentum_10', 'momentum_20'
        ]

        # Buat fitur teknikal
        # 1. Price range (high - low)
        df['price_range'] = df['high'] - df['low']
        
        # 2. Body size (close - open) abs value
        df['body_size'] = abs(df['price'] - df['open'])
        
        # 3. Price position in range (0 = bottom, 1 = top)
        df['price_position'] = (df['price'] - df['low']) / df['price_range'].replace(0, 1)
        
        # 4. Normalized volume
        if 'volume' in df.columns:
            df['norm_volume'] = df['volume'] / df['volume'].rolling(window=20, min_periods=1).mean()
        else:
            df['norm_volume'] = 0.0
        
        # 5. Simple Moving Averages
        df['sma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
        df['sma_10'] = df['price'].rolling(window=10, min_periods=1).mean()
        df['sma_20'] = df['price'].rolling(window=20, min_periods=1).mean()
        
        # 6. Distance from SMAs
        df['dist_sma_5'] = (df['price'] - df['sma_5']) / df['price']
        df['dist_sma_10'] = (df['price'] - df['sma_10']) / df['price']
        df['dist_sma_20'] = (df['price'] - df['sma_20']) / df['price']
        
        # 7. Momentum
        df['momentum_5'] = df['price'].pct_change(periods=5)
        df['momentum_10'] = df['price'].pct_change(periods=10)
        df['momentum_20'] = df['price'].pct_change(periods=20)
        
        # Isi NaN values
        df.fillna(0, inplace=True)
        
        # Ambil fitur dan timestamps
        features = df[feature_columns].values
        timestamps = df['timestamp'].values
        prices = df['price'].values
        
        # Buat array untuk X dan y
        X = []
        y = []
        ts = []
        
        # Untuk setiap time step yang cukup untuk prediction horizon
        for i in range(len(features) - prediction_horizon):
            # Ambil fitur saat ini
            X.append(features[i])
            
            # Target adalah log-return dari harga saat ini ke harga di masa depan
            future_log_return = np.log(prices[i + prediction_horizon] / prices[i])
            y.append(future_log_return)
            
            # Simpan timestamp
            ts.append(timestamps[i])
        
        # Konversi ke numpy arrays
        X = np.array(X)
        y = np.array(y)
        ts = np.array(ts)
        
        # Buat dan fit scaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Prepared log-return data with 17 features: X shape = {X_scaled.shape}, y shape = {y.shape}")
        
        return X_scaled, y, scaler, ts

    except Exception as e:
        logger.error(f"Error preparing log-return data: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None
