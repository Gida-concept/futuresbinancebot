import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
import logging
import pickle
import os
import config
from utils.indicators import get_feature_columns

logger = logging.getLogger(__name__)


class MLModel:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.is_trained = False
        self.accuracy = 0.0
        self.feature_columns = []

    def train(self, df: pd.DataFrame) -> bool:
        """Train XGBoost model with balanced classes"""
        try:
            if len(df) < 100:
                logger.warning(f"Insufficient data for training {self.symbol}. "
                               f"Need at least 100 rows, have {len(df)}")
                return False

            logger.info(f"Training model for {self.symbol} with {len(df)} samples")

            # Get features and target
            self.feature_columns = get_feature_columns(df)
            X = df[self.feature_columns].values
            y = df['target'].values

            # Check class distribution
            unique, counts = np.unique(y, return_counts=True)
            class_distribution = dict(zip(unique, counts))
            long_pct = (class_distribution.get(1, 0) / len(y)) * 100
            short_pct = (class_distribution.get(0, 0) / len(y)) * 100

            logger.info(f"   Class distribution: LONG: {long_pct:.1f}% | SHORT: {short_pct:.1f}%")

            # Calculate scale_pos_weight to balance classes
            if 0 in class_distribution and 1 in class_distribution:
                scale_pos_weight = class_distribution[0] / class_distribution[1]
            else:
                scale_pos_weight = 1.0

            logger.info(f"   Using scale_pos_weight: {scale_pos_weight:.2f} to balance classes")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=False
            )

            # Train XGBoost with class balancing
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,  # Balance classes
                random_state=42,
                eval_metric='logloss'
            )

            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )

            # Evaluate
            y_pred = self.model.predict(X_test)
            self.accuracy = accuracy_score(y_test, y_pred)

            # Check prediction distribution
            pred_unique, pred_counts = np.unique(y_pred, return_counts=True)
            pred_distribution = dict(zip(pred_unique, pred_counts))
            pred_long_pct = (pred_distribution.get(1, 0) / len(y_pred)) * 100
            pred_short_pct = (pred_distribution.get(0, 0) / len(y_pred)) * 100

            logger.info(f"Model trained for {self.symbol}. Accuracy: {self.accuracy:.4f}")
            logger.info(f"   Prediction distribution: LONG: {pred_long_pct:.1f}% | SHORT: {pred_short_pct:.1f}%")
            logger.debug(f"\n{classification_report(y_test, y_pred)}")

            self.is_trained = True
            self.save_model()

            return True

        except Exception as e:
            logger.error(f"Error training model for {self.symbol}: {e}")
            return False

    def predict(self, features: np.ndarray) -> tuple:
        """
        Predict next candle direction
        Returns: (prediction, probability)
        """
        try:
            if not self.is_trained or self.model is None:
                logger.warning(f"Model for {self.symbol} not trained yet")
                return None, None

            if features is None or len(features) == 0:
                logger.warning(f"No features provided for prediction")
                return None, None

            # Predict
            prediction = self.model.predict(features)[0]
            probability = self.model.predict_proba(features)[0]

            # Get probability of the predicted class
            pred_prob = probability[prediction]

            logger.debug(f"Prediction for {self.symbol}: {prediction} "
                         f"(prob: {pred_prob:.4f})")

            return prediction, pred_prob

        except Exception as e:
            logger.error(f"Error making prediction for {self.symbol}: {e}")
            return None, None

    def should_retrain(self, candle_count: int) -> bool:
        """Check if model should be retrained"""
        return candle_count % config.RETRAIN_INTERVAL == 0

    def save_model(self):
        """Save model to disk"""
        try:
            model_path = os.path.join(config.MODEL_DIR, f"{self.symbol}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'feature_columns': self.feature_columns,
                    'accuracy': self.accuracy
                }, f)
            logger.info(f"Model saved for {self.symbol}")
        except Exception as e:
            logger.error(f"Error saving model for {self.symbol}: {e}")

    def load_model(self) -> bool:
        """Load model from disk"""
        try:
            model_path = os.path.join(config.MODEL_DIR, f"{self.symbol}_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.feature_columns = data['feature_columns']
                    self.accuracy = data['accuracy']
                    self.is_trained = True
                logger.info(f"Model loaded for {self.symbol}. Accuracy: {self.accuracy:.4f}")
                return True
        except Exception as e:
            logger.error(f"Error loading model for {self.symbol}: {e}")
        return False