import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to dataframe
    """
    df = df.copy()

    # EMA
    df['ema_9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
    df['ema_21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()

    # RSI
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()

    # MACD
    macd = MACD(close=df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = bollinger.bollinger_wband()

    # ATR
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'],
                                 close=df['close'], window=14).average_true_range()

    # Price changes
    df['price_change'] = df['close'].pct_change()
    df['volume_change'] = df['volume'].pct_change()

    # Price position relative to range (with safety check)
    price_range = df['high'] - df['low']
    price_range = price_range.replace(0, np.nan)  # Avoid division by zero
    df['price_position'] = (df['close'] - df['low']) / price_range

    # Replace inf and -inf with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN values using modern pandas syntax
    df = df.ffill().bfill().fillna(0)  # ✅ NEW: Modern syntax

    return df


def create_features(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Create features for ML model including lagged values
    """
    df = add_technical_indicators(df)

    # Add lagged features
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'price_change']

    for col in feature_cols:
        for lag in range(1, lookback + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Create target: next candle up (1) or down (0)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

    # Replace any remaining inf/-inf/nan
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill().fillna(0)  # ✅ NEW: Modern syntax

    # Drop NaN values
    df = df.dropna()

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Get list of feature columns (exclude OHLCV and target)
    """
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'target']
    return [col for col in df.columns if col not in exclude_cols]