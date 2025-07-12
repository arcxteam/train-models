import os

# Base paths
APP_BASE_PATH = os.getenv("APP_BASE_PATH", default=os.getcwd())
DATA_BASE_PATH = os.path.join(APP_BASE_PATH, "data")

# Models directory
MODELS_DIR = os.path.join(APP_BASE_PATH, 'models')
if os.environ.get('MODELS_DIR'):
    MODELS_DIR = os.environ.get('MODELS_DIR')

# Tiingo API Configuration
TIINGO_API_TOKEN = os.environ.get('TIINGO_API_TOKEN', 'xxxa99de96faaf30fc292a9c5b785974efa85708646')
TIINGO_DATA_DIR = os.path.join(DATA_BASE_PATH, 'tiingo_data')
TIINGO_CACHE_TTL = int(os.environ.get('TIINGO_CACHE_TTL', 150))
