import os

APP_BASE_PATH = os.getenv("APP_BASE_PATH", default=os.getcwd())
DATA_BASE_PATH = os.path.join(APP_BASE_PATH, "data")
DATABASE_PATH = os.environ.get('DATABASE_PATH', 'prices.db')
WORKER_TYPE = int(os.environ.get('WORKER_TYPE', 1))
BLOCK_TIME_SECONDS = 5
ALLORA_VALIDATOR_API_URL = os.environ.get('ALLORA_VALIDATOR_API_URL', 'http://localhost:1317/')

# Endpoint for querying the latest block
URL_QUERY_LATEST_BLOCK = "cosmos/base/tendermint/v1beta1/blocks/latest"

# API Keys
CGC_API_KEY = os.environ.get('CGC_API_KEY', '')

# Tiingo API Configuration
TIINGO_API_TOKEN = os.environ.get('TIINGO_API_TOKEN', 'a99de9uwyww628633d785974efa85708646')
TIINGO_CACHE_DIR = os.path.join(DATA_BASE_PATH, 'tiingo_cache')
TIINGO_CACHE_TTL = 10 * 60  # 10 minutes

# OKX Configuration
OKX_CACHE_DIR = os.path.join(DATA_BASE_PATH, 'okx_cache')
