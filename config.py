import os
from dotenv import load_dotenv

load_dotenv()

# Binance API Configuration
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', 'your_api_key')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', 'your_api_secret')
BINANCE_TESTNET = True

# Add this to config.py
TRADE_COOLDOWN_SECONDS = 60  # Wait 60 seconds between trades

# Trading Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
TIMEFRAME = '5m'
LEVERAGE = 10

# Safety buffer for TP/SL (prevents immediate trigger errors)
TP_SL_BUFFER_PERCENT = 0.1       # Add 0.1% buffer to avoid edge cases

# Risk Management (POSITION-based - Recommended)
POSITION_SIZE_PERCENT = 0.5      # Use 0.9% of account for each trade
TAKE_PROFIT_PERCENT = 2        # 2% profit on position size
STOP_LOSS_PERCENT = 1          # 1% loss on position size
MAX_ACTIVE_TRADES = 1

# Example with $5000 account:
# - Position size: $45 (0.9% of $5000)
# - With 10x leverage: $450 position
# - Take profit: $9 (2% of $450) = 0.18% of account
# - Stop loss: $4.50 (1% of $450) = 0.09% of account
# - Risk/Reward: 1:2 (good ratio)

# XGBoost Model Configuration
LOOKBACK_CANDLES = 100
RETRAIN_INTERVAL = 500
PREDICTION_THRESHOLD = 0.6
HISTORICAL_DAYS = 60

# Telegram Configuration
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', 'your_bot_token')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', 'your_chat_id')
SEND_HEARTBEAT = True
HEARTBEAT_INTERVAL = 60

# Logging
LOG_FILE = 'logs/bot.log'
LOG_LEVEL = 'INFO'

# Paths
MODEL_DIR = 'models/'
DATA_DIR = 'data/'

# WebSocket Configuration
WS_RECONNECT_DELAY = 5
WS_PING_INTERVAL = 30

# Create directories
os.makedirs('logs', exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)