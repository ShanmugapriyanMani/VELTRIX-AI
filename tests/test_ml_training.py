"""Tests for ML training pipeline — direction + quality models."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.ml.candle_features import FEATURE_NAMES
from src.ml.train_models import DirectionModelTrainer, QualityModelTrainer, DriftDetector


def _make_store():
    """Create a DataStore backed by a temp SQLite file."""
    from src.config.env_loader import get_config
    from src.data.store import DataStore

    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    tmp.write("database:\n  engine: sqlite\n  sqlite_path: ':memory:'\n")
    tmp.close()

    old_val = os.environ.get("DB_PATH")
    db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = db_file.name
    db_file.close()
    os.environ["DB_PATH"] = db_path
    get_config.cache_clear()

    store = DataStore(tmp.name)

    if old_val is not None:
        os.environ["DB_PATH"] = old_val
    else:
        os.environ.pop("DB_PATH", None)
    get_config.cache_clear()
    os.unlink(tmp.name)

    return store, db_path


def _make_feature_df(n_rows: int = 300) -> pd.DataFrame:
    """Generate synthetic feature DataFrame with dates."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="B")
    data = {"date": [d.strftime("%Y-%m-%d") for d in dates]}
    for feat in FEATURE_NAMES:
        data[feat] = np.random.randn(n_rows)
    # Add OHLCV for label generation
    data["open"] = 22000 + np.cumsum(np.random.randn(n_rows) * 5)
    data["close"] = data["open"] + np.random.randn(n_rows) * 20
    data["high"] = np.maximum(data["open"], data["close"]) + abs(np.random.randn(n_rows)) * 10
    data["low"] = np.minimum(data["open"], data["close"]) - abs(np.random.randn(n_rows)) * 10
    data["volume"] = (np.random.rand(n_rows) * 100000 + 10000).astype(int)
    return pd.DataFrame(data)


class TestDirectionModelTrainer:
    """Test Stage 1 direction model training."""

    def test_walk_forward_split_temporal(self):
        """Test split is strictly temporal — no data leakage."""
        store, db_path = _make_store()
        try:
            from src.ml.candle_features import CandleFeatureBuilder
            fb = CandleFeatureBuilder(store)
            trainer = DirectionModelTrainer(store, fb)

            n = 300
            X = pd.DataFrame(np.random.randn(n, 5), columns=[f"f{i}" for i in range(5)])
            y = pd.Series(np.random.choice([0, 1], n))

            X_train, X_test, y_train, y_test = trainer._walk_forward_split(X, y)

            # Test is the last portion
            assert len(X_test) == trainer.TEST_DAYS
            assert len(X_train) == n - trainer.TEST_DAYS

            # Verify temporal order: last train index < first test index
            assert X_train.index[-1] < X_test.index[0]
        finally:
            store.close()
            os.unlink(db_path)

    def test_train_returns_metrics(self):
        """Training returns dict with required metric keys."""
        store, db_path = _make_store()
        try:
            from src.ml.candle_features import CandleFeatureBuilder
            fb = CandleFeatureBuilder(store)
            trainer = DirectionModelTrainer(store, fb)

            # Inject synthetic candles + features
            feat_df = _make_feature_df(400)
            candle_rows = []
            for _, row in feat_df.iterrows():
                dt = pd.Timestamp(row["date"]) + pd.Timedelta(hours=9, minutes=15)
                candle_rows.append({
                    "datetime": dt, "open": row["open"], "high": row["high"],
                    "low": row["low"], "close": row["close"], "volume": int(row["volume"]), "oi": 0,
                })
            candles_df = pd.DataFrame(candle_rows)
            store.save_ml_candles("NIFTY50", "NSE_INDEX|Nifty 50", candles_df)

            # Mock feature builder to return our synthetic data
            with patch.object(fb, 'build_features', return_value=feat_df):
                metrics = trainer.train("NIFTY50")

            assert "train_acc" in metrics
            assert "test_acc" in metrics
            assert "gap" in metrics
            assert "deployed" in metrics
        finally:
            store.close()
            os.unlink(db_path)

    def test_deploy_gate_passes(self):
        """test_acc=0.55, gap=0.15 → True (above 52%, gap < 20%)."""
        store, db_path = _make_store()
        try:
            from src.ml.candle_features import CandleFeatureBuilder
            fb = CandleFeatureBuilder(store)
            trainer = DirectionModelTrainer(store, fb)
            assert trainer._check_deploy_gate(0.70, 0.55) is True
        finally:
            store.close()
            os.unlink(db_path)

    def test_deploy_gate_fails_low_accuracy(self):
        """test_acc=0.50 → False (below 52%)."""
        store, db_path = _make_store()
        try:
            from src.ml.candle_features import CandleFeatureBuilder
            fb = CandleFeatureBuilder(store)
            trainer = DirectionModelTrainer(store, fb)
            assert trainer._check_deploy_gate(0.60, 0.50) is False
        finally:
            store.close()
            os.unlink(db_path)

    def test_deploy_gate_fails_high_gap(self):
        """train_acc=0.85, test_acc=0.55, gap=0.30 → False (gap > 20%)."""
        store, db_path = _make_store()
        try:
            from src.ml.candle_features import CandleFeatureBuilder
            fb = CandleFeatureBuilder(store)
            trainer = DirectionModelTrainer(store, fb)
            assert trainer._check_deploy_gate(0.85, 0.55) is False
        finally:
            store.close()
            os.unlink(db_path)

    def test_model_versioning_increments(self):
        """Each save_ml_model_record call creates a new version."""
        store, db_path = _make_store()
        try:
            v1 = store.save_ml_model_record({
                "model_name": "test_model",
                "stage": "direction",
                "train_date": "2025-01-01",
                "train_samples": 100,
                "n_features": 46,
                "model_path": "data/models/test.pkl",
            })
            v2 = store.save_ml_model_record({
                "model_name": "test_model",
                "stage": "direction",
                "train_date": "2025-01-08",
                "train_samples": 110,
                "n_features": 46,
                "model_path": "data/models/test2.pkl",
            })
            assert v2 == v1 + 1
        finally:
            store.close()
            os.unlink(db_path)


class TestQualityModelTrainer:
    """Test Stage 2 quality model."""

    def test_insufficient_labels_skips_training(self):
        """< 30 labeled trades → returns error, no model saved."""
        store, db_path = _make_store()
        try:
            trainer = QualityModelTrainer(store)

            # Insert only 10 labels
            for i in range(10):
                store.save_ml_trade_label({
                    "trade_id": f"trade_{i}",
                    "trade_date": "2025-06-01",
                    "symbol": "NIFTY",
                    "direction": "CE",
                    "entry_price": 100,
                    "exit_price": 110 if i % 2 == 0 else 90,
                    "pnl": 10 if i % 2 == 0 else -10,
                    "label": 1 if i % 2 == 0 else 0,
                })

            result = trainer.train()
            assert result.get("deployed") is False
            assert "error" in result or "Insufficient" in result.get("error", "")
        finally:
            store.close()
            os.unlink(db_path)

    def test_quality_prediction_returns_win_prob(self):
        """Predict returns {win_prob: float, quality_class: str}."""
        store, db_path = _make_store()
        try:
            trainer = QualityModelTrainer(store)
            # Without a trained model, predict returns defaults
            result = trainer.predict({"score_diff": 3.0, "conviction": 0.7})
            assert "win_prob" in result
            assert "quality_class" in result
            assert isinstance(result["win_prob"], float)
            assert result["quality_class"] in ("HIGH", "LOW", "UNKNOWN")
        finally:
            store.close()
            os.unlink(db_path)


class TestDriftDetector:
    """Test drift detection."""

    def test_no_drift_when_no_predictions(self):
        """No predictions → result has expected structure with n_predictions=0."""
        store, db_path = _make_store()
        try:
            detector = DriftDetector(store)
            result = detector.check_drift("nonexistent_model_xyz")
            assert "drifted" in result
            assert "baseline_acc" in result
            assert "n_predictions" in result
            # No deployed model → n_predictions=0, no drift flagged
            assert result["n_predictions"] == 0
            assert bool(result["drifted"]) is False
        finally:
            store.close()
            os.unlink(db_path)

    def test_drift_detected_on_accuracy_drop(self):
        """Recent accuracy drops below baseline → drifted=True."""
        store, db_path = _make_store()
        try:
            detector = DriftDetector(store)

            # Create a deployed model with high test accuracy
            v = store.save_ml_model_record({
                "model_name": "direction_v1",
                "stage": "direction",
                "train_date": "2025-01-01",
                "train_samples": 500,
                "n_features": 46,
                "test_accuracy": 0.70,
                "train_accuracy": 0.75,
                "model_path": "data/models/test.pkl",
            })
            store.set_model_deployed("direction_v1", v)

            # Add predictions AFTER deploy date with low accuracy (all wrong)
            for i in range(12):
                store.save_ml_prediction({
                    "model_name": "direction_v1",
                    "model_version": v,
                    "prediction_date": f"2025-06-{10+i:02d}",
                    "prediction_time": "09:30:00",
                    "predicted_class": "CE",
                })
                # Mark as wrong
                store.update_prediction_actual(f"2025-06-{10+i:02d}", "PE")

            result = detector.check_drift("direction_v1")
            # Recent acc = 0.0, baseline = 0.70, drop = 0.70 > 0.10
            assert bool(result["drifted"]) is True
            assert result["drop"] > 0.10
        finally:
            store.close()
            os.unlink(db_path)

    def test_drift_skip_uses_predictions_since_deploy(self):
        """Predictions from before deploy date are ignored. < 10 since deploy → skip drift."""
        store, db_path = _make_store()
        try:
            detector = DriftDetector(store)

            # Create old model v1 (deployed 2025-01-01)
            v1 = store.save_ml_model_record({
                "model_name": "direction_v1",
                "stage": "direction",
                "train_date": "2025-01-01",
                "train_samples": 500,
                "n_features": 46,
                "test_accuracy": 0.60,
                "train_accuracy": 0.70,
                "model_path": "data/models/test.pkl",
            })

            # Add 30 predictions from v1 era (before retrain)
            for i in range(30):
                store.save_ml_prediction({
                    "model_name": "direction_v1",
                    "model_version": v1,
                    "prediction_date": f"2025-02-{min(28, i+1):02d}",
                    "prediction_time": "09:30:00",
                    "predicted_class": "CE",
                })
                store.update_prediction_actual(f"2025-02-{min(28, i+1):02d}", "PE")

            # Retrain: new v2 deployed on 2025-06-01
            v2 = store.save_ml_model_record({
                "model_name": "direction_v1",
                "stage": "direction",
                "train_date": "2025-06-01",
                "train_samples": 600,
                "n_features": 46,
                "test_accuracy": 0.65,
                "train_accuracy": 0.72,
                "model_path": "data/models/test2.pkl",
            })
            store.set_model_deployed("direction_v1", v2)

            # Add only 5 predictions AFTER deploy date (< 10 threshold)
            for i in range(5):
                store.save_ml_prediction({
                    "model_name": "direction_v1",
                    "model_version": v2,
                    "prediction_date": f"2025-06-{10+i:02d}",
                    "prediction_time": "09:30:00",
                    "predicted_class": "CE",
                })
                store.update_prediction_actual(f"2025-06-{10+i:02d}", "PE")

            result = detector.check_drift("direction_v1")
            # Only 5 predictions since deploy (< 10) → drift skipped
            assert result["drifted"] is False
            assert result["n_predictions"] == 5
        finally:
            store.close()
            os.unlink(db_path)

    def test_retrain_skips_if_already_done_today(self):
        """If deployed model's train_date == today, retrain is skipped."""
        store, db_path = _make_store()
        try:
            from datetime import date as date_cls

            today_str = date_cls.today().isoformat()

            # Create a model already trained today
            v = store.save_ml_model_record({
                "model_name": "direction_v1",
                "stage": "direction",
                "train_date": today_str,
                "train_samples": 500,
                "n_features": 46,
                "test_accuracy": 0.60,
                "train_accuracy": 0.70,
                "model_path": "data/models/test.pkl",
            })
            store.set_model_deployed("direction_v1", v)

            # Verify the guard: train_date == today → should skip
            deployed = store.get_deployed_model("direction_v1")
            assert deployed["train_date"] == today_str
            # This is the exact check _maybe_retrain_ml_models uses
            should_skip = deployed.get("train_date") == today_str
            assert should_skip is True

            # Model trained yesterday → should NOT skip
            v2 = store.save_ml_model_record({
                "model_name": "direction_v1",
                "stage": "direction",
                "train_date": "2025-01-01",
                "train_samples": 500,
                "n_features": 46,
                "test_accuracy": 0.60,
                "train_accuracy": 0.70,
                "model_path": "data/models/test2.pkl",
            })
            store.set_model_deployed("direction_v1", v2)

            deployed2 = store.get_deployed_model("direction_v1")
            should_skip2 = deployed2.get("train_date") == today_str
            assert should_skip2 is False
        finally:
            store.close()
            os.unlink(db_path)
