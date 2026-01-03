import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
import logging
import pickle
import os
import config
from utils.indicators import create_features, get_feature_columns

logger = logging.getLogger(__name__)


class DataManager:
    def __init__(self, symbol: str, client=None):
        self.symbol = symbol
        self.client = client
        self.candles = pd.DataFrame()
        self.features_df = pd.DataFrame()
        self.candle_count = 0

    def set_client(self, client):
        """Set Binance client"""
        self.client = client

    def fetch_historical_data(self, days: int = None) -> pd.DataFrame:
        """Fetch historical kline data from Binance"""
        if days is None:
            days = config.HISTORICAL_DAYS

        if self.client is None:
            logger.error(f"Client not initialized for {self.symbol}")
            return pd.DataFrame()

        try:
            logger.info(f"Fetching {days} days of historical data for {self.symbol}")

            # Calculate start time in milliseconds
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)

            # Convert to milliseconds timestamp (Binance expects this)
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)

            logger.debug(f"Fetching from {start_time} to {end_time}")

            # Fetch klines
            klines = self.client.futures_klines(
                symbol=self.symbol,
                interval=config.TIMEFRAME,
                startTime=start_ms,
                endTime=end_ms,
                limit=1000
            )

            if not klines:
                logger.error(f"No data returned for {self.symbol}")
                return pd.DataFrame()

            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])

            # Convert to appropriate types
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # Keep only necessary columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            logger.info(f"✓ Fetched {len(df)} candles for {self.symbol}")
            self.candles = df

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {self.symbol}: {e}")
            return pd.DataFrame()

    def add_candle(self, candle_data: dict):
        """Add new candle to dataframe"""
        try:
            new_candle = pd.DataFrame([{
                'timestamp': pd.to_datetime(candle_data['timestamp'], unit='ms'),
                'open': float(candle_data['open']),
                'high': float(candle_data['high']),
                'low': float(candle_data['low']),
                'close': float(candle_data['close']),
                'volume': float(candle_data['volume'])
            }])

            self.candles = pd.concat([self.candles, new_candle], ignore_index=True)
            self.candle_count += 1

            # Keep only recent candles to save memory
            max_candles = config.LOOKBACK_CANDLES * 3
            if len(self.candles) > max_candles:
                self.candles = self.candles.tail(max_candles).reset_index(drop=True)

            logger.debug(f"Added candle for {self.symbol}. Total: {len(self.candles)}")

        except Exception as e:
            logger.error(f"Error adding candle for {self.symbol}: {e}")

    def prepare_features(self) -> pd.DataFrame:
        """Prepare features for ML model"""
        try:
            if len(self.candles) < config.LOOKBACK_CANDLES:
                logger.warning(f"Not enough candles for {self.symbol}. "
                               f"Have {len(self.candles)}, need {config.LOOKBACK_CANDLES}")
                return pd.DataFrame()

            self.features_df = create_features(self.candles.copy(), lookback=5)
            logger.debug(f"Prepared {len(self.features_df)} feature rows for {self.symbol}")

            return self.features_df

        except Exception as e:
            logger.error(f"Error preparing features for {self.symbol}: {e}")
            return pd.DataFrame()

    def get_latest_features(self) -> np.ndarray:
        """Get features for the latest candle for prediction"""
        try:
            if self.features_df.empty:
                self.prepare_features()

            if self.features_df.empty:
                return None

            feature_cols = get_feature_columns(self.features_df)
            latest = self.features_df[feature_cols].iloc[-1:].values

            return latest

        except Exception as e:
            logger.error(f"Error getting latest features for {self.symbol}: {e}")
            return None

    def get_current_price(self) -> float:
        """Get current close price"""
        if len(self.candles) > 0:
            return self.candles['close'].iloc[-1]
        return 0.0

    def save_data(self):
        """Save candles to disk"""
        try:
            filepath = os.path.join(config.DATA_DIR, f"{self.symbol}_candles.pkl")
            with open(filepath, 'wb') as f:
                pickle.dump(self.candles, f)
            logger.info(f"Saved data for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving data for {self.symbol}: {e}")

    def load_data(self):
        """Load candles from disk"""
        try:
            filepath = os.path.join(config.DATA_DIR, f"{self.symbol}_candles.pkl")
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    self.candles = pickle.load(f)
                logger.info(f"Loaded {len(self.candles)} candles for {self.symbol}")
                return True
        except Exception as e:
            logger.error(f"Error loading data for {self.symbol}: {e}")
        return False