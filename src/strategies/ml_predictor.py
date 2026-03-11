"""
Strategy 4: ML Prediction with Alternative Data.

Model: LightGBM classifier for next-day direction.
Features (15): FII flows, VIX, PCR, delivery %, sector strength, technicals.
Training: Walk-forward — 252-day train, 5-day test, retrain weekly (Saturday).
Only signal if predict_proba > 0.62 to avoid noise.
Anti-overfit: max 15 features, early stopping, 5-fold TimeSeriesSplit.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, date
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from loguru import logger

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("lightgbm not installed. ML predictor will be disabled.")

try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from src.strategies.base import BaseStrategy, Signal, SignalDirection


class MLPredictorStrategy(BaseStrategy):
    """
    ML-based next-day direction predictor using LightGBM.

    Uses India-specific alternative data features that give structural edge
    over purely technical approaches.
    """

    def __init__(self, config_path: str = "config/strategies.yaml"):
        super().__init__("ml_predictor", config_path)

        self.model_path = self.config.get("model_path", "models/lgbm_latest.pkl")
        self.scaler_path = self.config.get("scaler_path", "models/scaler_latest.pkl")
        self.confidence_threshold = self.config.get("confidence_threshold", 0.62)
        self.max_features = self.config.get("max_features", 15)
        self.feature_names = self.config.get("features", [])

        # Label thresholds
        label_cfg = self.config.get("label", {})
        self.up_threshold = label_cfg.get("up_threshold", 0.005)
        self.down_threshold = label_cfg.get("down_threshold", -0.005)

        # Training config
        train_cfg = self.config.get("training", {})
        self.train_window = train_cfg.get("train_window", 252)
        self.test_window = train_cfg.get("test_window", 5)
        self.n_splits = train_cfg.get("n_splits", 5)
        self.early_stopping = train_cfg.get("early_stopping_rounds", 50)
        self.min_trades = train_cfg.get("min_trades_significance", 200)

        # LightGBM params
        self.lgbm_params = self.config.get("lightgbm_params", {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "is_unbalance": True,        # Handle class imbalance (61% neutral)
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_estimators": 500,
            "verbose": -1,
        })

        # Model state
        self._model: Optional[Any] = None
        self._scaler: Optional[Any] = None
        self._feature_importance: dict[str, float] = {}
        self._last_train_date: Optional[date] = None
        self._train_metrics: dict[str, Any] = {}

        # Try loading saved model
        self._load_model()

    def _load_model(self) -> bool:
        """Load saved model and scaler from disk."""
        if not LIGHTGBM_AVAILABLE:
            return False

        # V9.2: ML permanently disabled — skip loading stale models
        from src.config.env_loader import get_config
        if not get_config().ML_ENABLED:
            return False

        model_file = Path(self.model_path)
        scaler_file = Path(self.scaler_path)

        if model_file.exists() and scaler_file.exists():
            try:
                with open(model_file, "rb") as f:
                    self._model = pickle.load(f)
                with open(scaler_file, "rb") as f:
                    self._scaler = pickle.load(f)
                logger.info(f"Loaded ML model from {model_file}")
                return True
            except Exception as e:
                logger.error(f"Failed to load model: {e}")

        return False

    def _save_model(self) -> None:
        """Save model and scaler to disk."""
        model_dir = Path(self.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(self.model_path, "wb") as f:
            pickle.dump(self._model, f)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self._scaler, f)
        logger.info(f"Saved ML model to {self.model_path}")

    def update(self, data: dict[str, Any]) -> None:
        """Update with latest feature data."""
        pass  # ML predictor uses batch features, no tick-level state

    def generate_signals(self, data: dict[str, Any]) -> list[Signal]:
        """
        Generate ML-based prediction signals.

        Expected data keys:
        - features_df: DataFrame with all features (per symbol)
        - stock_universe: dict of {symbol: {price, atr, ...}}
        - regime: str
        """
        signals = []
        regime = data.get("regime", "")

        if not self.is_active_in_regime(regime):
            return signals

        if not LIGHTGBM_AVAILABLE or self._model is None:
            logger.debug("ML model not available, skipping prediction")
            return signals

        features_dict = data.get("features_dict", {})  # {symbol: DataFrame}
        stock_universe = data.get("stock_universe", {})

        for symbol, feat_df in features_dict.items():
            signal = self._predict_single(symbol, feat_df, stock_universe, regime)
            if signal:
                signals.append(signal)

        self._signals_history.extend(signals)

        if signals:
            logger.info(
                f"ML_PREDICTOR: {len(signals)} signals generated | "
                f"Model trained: {self._last_train_date}"
            )

        return signals

    def _predict_single(
        self,
        symbol: str,
        feat_df: pd.DataFrame,
        stock_universe: dict,
        regime: str,
    ) -> Optional[Signal]:
        """Generate prediction for a single stock."""
        if feat_df.empty:
            return None

        # Select features
        available = [f for f in self.feature_names if f in feat_df.columns]
        if len(available) < 5:
            return None

        # Get latest row
        latest = feat_df[available].iloc[-1:].copy()

        # Handle NaNs
        if latest.isna().any().any():
            latest = latest.fillna(0)

        # Scale features
        if self._scaler is not None:
            try:
                latest_scaled = pd.DataFrame(
                    self._scaler.transform(latest),
                    columns=available,
                )
            except Exception:
                latest_scaled = latest
        else:
            latest_scaled = latest

        # Predict
        try:
            probas = self._model.predict_proba(latest_scaled)[0]
        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return None

        # Class mapping: 0 = down (-1), 1 = neutral (0), 2 = up (+1)
        prob_down = probas[0]
        prob_neutral = probas[1]
        prob_up = probas[2]

        # Get stock info
        stock_info = stock_universe.get(symbol, {})
        price = stock_info.get("price", 0)
        atr = stock_info.get("atr", price * 0.02 if price > 0 else 0)

        if price <= 0:
            return None

        # ── Generate signal if confident enough ──
        if prob_up > self.confidence_threshold:
            direction = SignalDirection.BUY
            confidence = float(prob_up)
            score = float(prob_up - prob_down)

        elif prob_down > self.confidence_threshold:
            direction = SignalDirection.SELL
            confidence = float(prob_down)
            score = float(prob_down - prob_up) * -1

        else:
            return None  # Not confident enough

        # Stop loss and take profit
        if direction == SignalDirection.BUY:
            sl = price - 1.5 * atr
            tp = price + 3.0 * atr
        else:
            sl = price + 1.5 * atr
            tp = price - 3.0 * atr

        return Signal(
            strategy=self.name,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            score=score,
            price=price,
            stop_loss=sl,
            take_profit=tp,
            hold_days=1,
            regime=regime,
            features={
                "prob_up": round(prob_up, 4),
                "prob_down": round(prob_down, 4),
                "prob_neutral": round(prob_neutral, 4),
                "top_features": self._get_top_features(available, latest),
            },
            notes=(
                f"ML: P(up)={prob_up:.3f}, P(down)={prob_down:.3f}. "
                f"{'BUY' if direction == SignalDirection.BUY else 'SELL'} "
                f"confidence={confidence:.3f}"
            ),
        )

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        validate: bool = True,
    ) -> dict[str, Any]:
        """
        Train the LightGBM model with walk-forward validation.

        Args:
            X: Feature matrix
            y: Target labels (-1, 0, 1)
            validate: Whether to run cross-validation

        Returns:
            Training metrics dict
        """
        if not LIGHTGBM_AVAILABLE or not SKLEARN_AVAILABLE:
            logger.error("LightGBM or sklearn not available for training")
            return {"error": "dependencies missing"}

        logger.info(
            f"Training ML model: {len(X)} samples, {len(X.columns)} features"
        )

        # Map labels: -1→0, 0→1, 1→2 for LightGBM multiclass
        y_mapped = y.map({-1: 0, 0: 1, 1: 2})

        # ── Cross-validation ──
        cv_scores = []
        if validate:
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train_cv = X.iloc[train_idx]
                y_train_cv = y_mapped.iloc[train_idx]
                X_val_cv = X.iloc[val_idx]
                y_val_cv = y_mapped.iloc[val_idx]

                # Scale (keep feature names to avoid sklearn warnings)
                scaler_cv = StandardScaler()
                X_train_scaled = pd.DataFrame(
                    scaler_cv.fit_transform(X_train_cv),
                    columns=X_train_cv.columns, index=X_train_cv.index,
                )
                X_val_scaled = pd.DataFrame(
                    scaler_cv.transform(X_val_cv),
                    columns=X_val_cv.columns, index=X_val_cv.index,
                )

                model_cv = lgb.LGBMClassifier(**self.lgbm_params)
                model_cv.fit(
                    X_train_scaled,
                    y_train_cv,
                    eval_set=[(X_val_scaled, y_val_cv)],
                    callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
                )

                y_pred = model_cv.predict(X_val_scaled)
                acc = accuracy_score(y_val_cv, y_pred)
                cv_scores.append(acc)
                logger.debug(f"CV Fold {fold + 1}: accuracy={acc:.4f}")

            mean_cv = np.mean(cv_scores)
            logger.info(
                f"Cross-validation: mean accuracy={mean_cv:.4f} ± {np.std(cv_scores):.4f}"
            )

        # ── Final model on full data ──
        self._scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            self._scaler.fit_transform(X),
            columns=X.columns, index=X.index,
        )

        self._model = lgb.LGBMClassifier(**self.lgbm_params)

        # Split last test_window for final validation
        if len(X) > self.train_window + self.test_window:
            train_end = len(X) - self.test_window
            X_train = X_scaled.iloc[:train_end]
            y_train = y_mapped.iloc[:train_end]
            X_test = X_scaled.iloc[train_end:]
            y_test = y_mapped.iloc[train_end:]

            self._model.fit(
                X_train,
                y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)],
            )

            y_pred = self._model.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
        else:
            self._model.fit(X_scaled, y_mapped)
            test_acc = np.mean(cv_scores) if cv_scores else 0

        # ── Feature importance ──
        importance = self._model.feature_importances_
        self._feature_importance = dict(
            sorted(
                zip(X.columns, importance),
                key=lambda x: x[1],
                reverse=True,
            )
        )

        # Save model
        self._last_train_date = date.today()
        self._save_model()

        # Metrics
        self._train_metrics = {
            "train_date": date.today().isoformat(),
            "n_samples": len(X),
            "n_features": len(X.columns),
            "cv_accuracy_mean": round(float(np.mean(cv_scores)), 4) if cv_scores else 0,
            "cv_accuracy_std": round(float(np.std(cv_scores)), 4) if cv_scores else 0,
            "test_accuracy": round(float(test_acc), 4),
            "feature_importance": {
                k: round(float(v), 4)
                for k, v in list(self._feature_importance.items())[:10]
            },
            "class_distribution": dict(y.value_counts()),
        }

        logger.info(
            f"ML model trained: test_acc={test_acc:.4f}, "
            f"features={len(X.columns)}, samples={len(X)}"
        )

        # Log top features
        for feat, imp in list(self._feature_importance.items())[:5]:
            logger.info(f"  Feature: {feat} = {imp:.4f}")

        return self._train_metrics

    def needs_retraining(self) -> bool:
        """Check if the model needs retraining (Saturday schedule)."""
        today = date.today()

        if self._model is None:
            return True

        retrain_day = self.config.get("training", {}).get("retrain_day", "Saturday")
        day_map = {
            "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
            "Friday": 4, "Saturday": 5, "Sunday": 6,
        }

        if today.weekday() != day_map.get(retrain_day, 5):
            return False

        if self._last_train_date == today:
            return False

        return True

    def _get_top_features(
        self, feature_names: list[str], values: pd.DataFrame
    ) -> dict[str, float]:
        """Get top contributing features for this prediction."""
        if not self._feature_importance:
            return {}

        top = {}
        for feat in feature_names[:5]:
            if feat in self._feature_importance and feat in values.columns:
                top[feat] = float(values[feat].iloc[0])
        return top

    @property
    def model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "has_model": self._model is not None,
            "last_train_date": self._last_train_date.isoformat() if self._last_train_date else None,
            "n_features": len(self._feature_importance),
            "confidence_threshold": self.confidence_threshold,
            "train_metrics": self._train_metrics,
            "feature_importance": dict(list(self._feature_importance.items())[:10]),
        }
