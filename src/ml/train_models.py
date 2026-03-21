"""
ML Training Pipeline — XGBoost direction + quality models.

Stage 1: Direction Model (CE/PE 2-class from 5-min candle features)
  - XGBoost binary classifier
  - Walk-forward: train on all except last 3 months, test on last 3 months
  - Deploy gate: test_acc > 52%, gap < 20%

Stage 1b: Separate PE/CE Binary Models
  - pe_direction_v1: "Will NIFTY close lower?" (binary, 1=yes PE trade)
  - ce_direction_v1: "Will NIFTY close higher?" (binary, 1=yes CE trade)
  - Each model independently trained with 0.2% threshold
  - predict_direction_v2(): combines both models' probabilities
  - Deploy gate: test_acc > 52%, gap < 15%

Stage 2: Trade Quality Model (WIN/LOSS from trade features)
  - XGBoost binary classifier
  - Minimum 30 labeled trades required
  - Deploy gate: test_acc > 58%, gap < 18%

Model files saved to data/models/ (separate from old LightGBM in models/)
"""

from __future__ import annotations

import json
import pickle
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from src.data.store import DataStore
from src.ml.candle_features import CandleFeatureBuilder, FEATURE_NAMES, PE_FEATURE_NAMES, CE_FEATURE_NAMES


class DirectionModelTrainer:
    """Stage 1: Train direction prediction model."""

    MODEL_NAME = "direction_v1"
    MODEL_DIR = Path("data/models")
    DEPLOY_GATE = {"min_test_acc": 0.52, "max_gap": 0.20}
    TEST_DAYS = 63  # ~3 months of trading days

    def __init__(self, store: DataStore, feature_builder: CandleFeatureBuilder):
        self.store = store
        self.fb = feature_builder
        self.model = None
        self.scaler = None
        self.model_version: int = 0
        self.feature_names: list[str] = FEATURE_NAMES

    def train(self, symbol: str = "NIFTY50") -> dict:
        """
        Full training pipeline.

        Returns: {train_acc, test_acc, gap, deployed, model_version, class_report}
        """
        if not XGB_AVAILABLE:
            return {"error": "xgboost not installed", "deployed": False}

        # Build features
        features_df = self.fb.build_features(symbol)
        if features_df.empty or len(features_df) < 200:
            return {"error": f"Insufficient data: {len(features_df)} rows", "deployed": False}

        # Build daily aggregation for labels
        candles = self.store.get_ml_candles(symbol)
        if candles.empty:
            return {"error": "No candles for label generation", "deployed": False}

        candles["datetime"] = pd.to_datetime(
            candles["datetime"].astype(str).str.replace(r'\+\d{2}:\d{2}$', '', regex=True),
            format='%Y-%m-%d %H:%M:%S'
        )
        daily = self.fb._aggregate_daily(candles)
        labels = self.fb.compute_direction_labels(daily)

        # Align features and labels by date
        features_df = features_df.reset_index(drop=True)
        label_df = daily[["date"]].copy()
        label_df["label"] = labels

        merged = features_df.merge(label_df, on="date", how="inner")
        merged = merged.dropna(subset=["label"])
        merged = merged[merged["label"].notna()].reset_index(drop=True)

        # Remove last row (label uses next day which doesn't exist)
        merged = merged.iloc[:-1]

        if len(merged) < 200:
            return {"error": f"Insufficient aligned data: {len(merged)} rows", "deployed": False}

        # Extract X, y
        available_features = [f for f in self.feature_names if f in merged.columns]
        X = merged[available_features].fillna(0)
        # 2-class: CE=1, PE=0 (already in this format from compute_direction_labels)
        y = merged["label"].astype(int)

        # Walk-forward split
        X_train, X_test, y_train, y_test = self._walk_forward_split(X, y)

        logger.info(
            f"ML TRAIN direction: {len(X_train)} train, {len(X_test)} test, "
            f"{len(available_features)} features"
        )

        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        params = self._get_xgb_params()
        n_estimators = params.pop("n_estimators", 200)

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=20,
            **params,
        )
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )
        logger.debug(f"XGB stopped at {self.model.best_iteration} rounds (of {n_estimators})")

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        gap = train_acc - test_acc

        # Deploy gate
        gate_passed = self._check_deploy_gate(train_acc, test_acc)

        # Save model record (paths filled after we get the version)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        version = self.store.save_ml_model_record({
            "model_name": self.MODEL_NAME,
            "model_type": "xgboost",
            "stage": "direction",
            "train_date": date.today().isoformat(),
            "train_samples": len(X_train),
            "n_features": len(available_features),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_test_gap": gap,
            "deployed": 1 if gate_passed else 0,
            "deploy_gate_passed": 1 if gate_passed else 0,
            "model_path": "",
            "scaler_path": "",
            "feature_list": available_features,
            "hyperparams": self._get_xgb_params(),
            "metrics_json": {
                "class_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
                "label_distribution": {
                    "CE": int((y == 1).sum()),
                    "PE": int((y == 0).sum()),
                },
            },
        })

        self.model_version = version

        if gate_passed:
            model_path = self.MODEL_DIR / f"{self.MODEL_NAME}_v{version}.pkl"
            scaler_path = self.MODEL_DIR / f"{self.MODEL_NAME}_scaler_v{version}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            # Update paths in DB record now that version is known
            self.store.update_ml_model_paths(
                self.MODEL_NAME, version, str(model_path), str(scaler_path)
            )
            self.store.set_model_deployed(self.MODEL_NAME, version)
            logger.info(
                f"ML DEPLOYED: {self.MODEL_NAME} v{version} "
                f"train={train_acc:.3f} test={test_acc:.3f} gap={gap:.3f}"
            )
        else:
            logger.warning(
                f"ML GATE FAILED: {self.MODEL_NAME} v{version} "
                f"train={train_acc:.3f} test={test_acc:.3f} gap={gap:.3f} "
                f"(need test>{self.DEPLOY_GATE['min_test_acc']}, gap<{self.DEPLOY_GATE['max_gap']})"
            )

        self.feature_names = available_features

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "gap": gap,
            "deployed": gate_passed,
            "model_version": version,
            "class_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def _walk_forward_split(
        self, X: pd.DataFrame, y: pd.Series, test_days: Optional[int] = None,
    ) -> tuple:
        """
        Temporal split: last test_days rows = test, rest = train.
        No shuffle. Strictly chronological.
        """
        n_test = test_days or self.TEST_DAYS
        n_test = min(n_test, len(X) // 4)  # At most 25% for test

        split_idx = len(X) - n_test
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def _check_deploy_gate(self, train_acc: float, test_acc: float) -> bool:
        """Deploy gate: test_acc > 52%, gap < 20%."""
        gap = train_acc - test_acc
        return test_acc > self.DEPLOY_GATE["min_test_acc"] and gap < self.DEPLOY_GATE["max_gap"]

    def _get_xgb_params(self) -> dict:
        """XGBoost hyperparameters for 2-class direction model."""
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 2,
            "learning_rate": 0.03,
            "n_estimators": 120,
            "min_child_weight": 30,
            "subsample": 0.65,
            "colsample_bytree": 0.6,
            "reg_alpha": 3.0,
            "reg_lambda": 3.0,
            "gamma": 0.5,
            "tree_method": "hist",
            "verbosity": 0,
        }

    def predict(self, features: dict) -> dict:
        """
        Predict direction from features dict.

        Returns: {predicted_class: "CE"/"PE", prob_ce, prob_pe}
        """
        if self.model is None or self.scaler is None:
            return {"predicted_class": "FLAT", "prob_ce": 0.50, "prob_pe": 0.50, "prob_flat": 0.0}

        feature_vals = [features.get(f, 0.0) for f in self.feature_names]
        X = pd.DataFrame([feature_vals], columns=self.feature_names)
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)[0]
        # 2-class: index 0=PE, index 1=CE
        prob_pe = float(probs[0])
        prob_ce = float(probs[1])

        return {
            "predicted_class": "CE" if prob_ce >= prob_pe else "PE",
            "prob_ce": prob_ce,
            "prob_pe": prob_pe,
            "prob_flat": 0.0,
        }

    def load_deployed_model(self) -> bool:
        """Load the currently deployed model from DB + disk."""
        record = self.store.get_deployed_model(self.MODEL_NAME)
        if not record:
            logger.debug(f"No deployed {self.MODEL_NAME} model found")
            return False

        model_path = Path(record["model_path"])
        scaler_path = Path(record["scaler_path"]) if record.get("scaler_path") else None

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            if scaler_path and scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            self.model_version = record["model_version"]
            if record.get("feature_list"):
                self.feature_names = json.loads(record["feature_list"])

            logger.info(
                f"ML LOADED: {self.MODEL_NAME} v{self.model_version} "
                f"test_acc={record['test_accuracy']:.1%}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load {self.MODEL_NAME}: {e}")
            return False


class BinaryDirectionTrainer:
    """
    Stage 1b: Separate PE and CE binary direction models.

    pe_direction_v1: "Will NIFTY drop fast today?" (1=yes, 0=no)
      - V2 label: 0.25% drop within any 6 consecutive 5-min candles (30 min)
    ce_direction_v1: "Will NIFTY rise fast today?" (1=yes, 0=no)
      - V2 label: 0.25% rise within any 6 consecutive 5-min candles (30 min)

    Both use intraday 30-min sliding window labels.
    """

    MODEL_DIR = Path("data/models")
    DEPLOY_GATE = {"min_test_acc": 0.52, "max_gap": 0.15, "min_precision": 0.40}
    PE_FILTER_THRESHOLD = 0.85  # Match options_buyer threshold for precision eval
    TEST_DAYS = 63
    MOVE_THRESHOLD = 0.002  # 0.2% — legacy (unused for CE/PE V2)
    PE_FAST_DROP = 0.0025   # 0.25% — PE drop in 6 candles (30 min)
    CE_FAST_RISE = 0.0025   # 0.25% — CE rise in 6 candles (30 min)

    def __init__(self, store: DataStore, feature_builder: CandleFeatureBuilder, side: str):
        """
        Args:
            side: "pe" or "ce"
        """
        assert side in ("pe", "ce"), f"side must be 'pe' or 'ce', got {side}"
        self.side = side
        self.MODEL_NAME = f"{side}_direction_v1"
        self.store = store
        self.fb = feature_builder
        self.model = None
        self.scaler = None
        self.model_version: int = 0
        # Direction-specific extended feature sets (51 base + 8 specific = 59)
        if side == "pe":
            self.feature_names: list[str] = PE_FEATURE_NAMES
        elif side == "ce":
            self.feature_names: list[str] = CE_FEATURE_NAMES
        else:
            self.feature_names: list[str] = FEATURE_NAMES

    def train(self, symbol: str = "NIFTY50") -> dict:
        """
        Train binary model for this side.

        PE label: 1 if next day has ≥0.25% drop in any 6 consecutive 5-min candles
        CE label: 1 if next day has ≥0.25% rise in any 6 consecutive 5-min candles

        Returns: {train_acc, test_acc, gap, deployed, model_version, precision, recall}
        """
        if not XGB_AVAILABLE:
            return {"error": "xgboost not installed", "deployed": False}

        # Reset feature list to full direction-specific set (load_deployed_model may have overwritten)
        if self.side == "pe":
            self.feature_names = PE_FEATURE_NAMES
        elif self.side == "ce":
            self.feature_names = CE_FEATURE_NAMES
        else:
            self.feature_names = FEATURE_NAMES

        # Build base features (51 standard)
        features_df = self.fb.build_features(symbol)
        if features_df.empty or len(features_df) < 200:
            return {"error": f"Insufficient data: {len(features_df)} rows", "deployed": False}

        # Build daily aggregation for custom binary labels
        candles = self.store.get_ml_candles(symbol)
        if candles.empty:
            return {"error": "No candles for label generation", "deployed": False}

        candles["datetime"] = pd.to_datetime(
            candles["datetime"].astype(str).str.replace(r'\+\d{2}:\d{2}$', '', regex=True),
            format='%Y-%m-%d %H:%M:%S'
        )
        daily = self.fb._aggregate_daily(candles)

        # Direction-specific features + labels
        if self.side == "pe":
            pe_feats = self.fb.compute_pe_specific_features(candles, daily)
            if not pe_feats.empty:
                features_df = features_df.merge(pe_feats, on="date", how="left")
                for col in pe_feats.columns:
                    if col != "date" and col in features_df.columns:
                        features_df[col] = features_df[col].fillna(0.0)
            labels = self._compute_pe_fast_drop_labels(candles, daily)
        elif self.side == "ce":
            ce_feats = self.fb.compute_ce_specific_features(candles, daily)
            if not ce_feats.empty:
                features_df = features_df.merge(ce_feats, on="date", how="left")
                for col in ce_feats.columns:
                    if col != "date" and col in features_df.columns:
                        features_df[col] = features_df[col].fillna(0.0)
            labels = self._compute_ce_fast_rise_labels(candles, daily)
        else:
            labels = self._compute_binary_labels(daily)

        # Align features and labels by date
        features_df = features_df.reset_index(drop=True)
        label_df = daily[["date"]].copy()
        label_df["label"] = labels

        merged = features_df.merge(label_df, on="date", how="inner")
        merged = merged.dropna(subset=["label"])
        merged = merged.iloc[:-1]  # Remove last row (label uses next day)

        if len(merged) < 200:
            return {"error": f"Insufficient aligned data: {len(merged)} rows", "deployed": False}

        # Extract X, y
        available_features = [f for f in self.feature_names if f in merged.columns]
        X = merged[available_features].fillna(0)
        y = merged["label"].astype(int)

        # Walk-forward split
        n_test = min(self.TEST_DAYS, len(X) // 4)
        split_idx = len(X) - n_test
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Handle class imbalance
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        scale_pos_weight = n_neg / max(n_pos, 1)

        logger.info(
            f"ML TRAIN {self.MODEL_NAME}: {len(X_train)} train, {len(X_test)} test, "
            f"{len(available_features)} features, pos_weight={scale_pos_weight:.2f}"
        )

        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train with binary-specific params
        params = self._get_xgb_params(scale_pos_weight)
        n_estimators = params.pop("n_estimators", 200)

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            early_stopping_rounds=20,
            **params,
        )
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            verbose=False,
        )

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        gap = train_acc - test_acc

        test_precision = precision_score(y_test, test_pred, zero_division=0)
        test_recall = recall_score(y_test, test_pred, zero_division=0)

        # Threshold-adjusted precision (matches live filter threshold)
        test_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        filter_threshold = self.PE_FILTER_THRESHOLD if self.side == "pe" else 0.50
        pred_at_threshold = (test_proba >= filter_threshold).astype(int)
        n_at_threshold = int(pred_at_threshold.sum())
        precision_at_threshold = precision_score(y_test, pred_at_threshold, zero_division=0)
        logger.info(
            f"PE_PRECISION_AT_{int(filter_threshold*100)}: "
            f"precision={precision_at_threshold:.1%} "
            f"({n_at_threshold} predictions at threshold {filter_threshold:.2f})"
        )

        # Deploy gate uses threshold-adjusted precision for PE
        deploy_precision = precision_at_threshold if self.side == "pe" else test_precision
        gate_passed = self._check_deploy_gate(train_acc, test_acc, deploy_precision)

        # Save model record
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        version = self.store.save_ml_model_record({
            "model_name": self.MODEL_NAME,
            "model_type": "xgboost",
            "stage": f"direction_{self.side}",
            "train_date": date.today().isoformat(),
            "train_samples": len(X_train),
            "n_features": len(available_features),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_test_gap": gap,
            "deployed": 1 if gate_passed else 0,
            "deploy_gate_passed": 1 if gate_passed else 0,
            "model_path": "",
            "scaler_path": "",
            "feature_list": available_features,
            "hyperparams": self._get_xgb_params(scale_pos_weight),
            "metrics_json": {
                "class_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
                "precision": test_precision,
                "precision_at_threshold": precision_at_threshold,
                "filter_threshold": filter_threshold,
                "n_at_threshold": n_at_threshold,
                "recall": test_recall,
                "label_distribution": {
                    "positive": int((y == 1).sum()),
                    "negative": int((y == 0).sum()),
                },
            },
        })

        self.model_version = version

        if gate_passed:
            model_path = self.MODEL_DIR / f"{self.MODEL_NAME}_v{version}.pkl"
            scaler_path = self.MODEL_DIR / f"{self.MODEL_NAME}_scaler_v{version}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            self.store.update_ml_model_paths(
                self.MODEL_NAME, version, str(model_path), str(scaler_path)
            )
            self.store.set_model_deployed(self.MODEL_NAME, version)
            logger.info(
                f"ML DEPLOYED: {self.MODEL_NAME} v{version} "
                f"train={train_acc:.3f} test={test_acc:.3f} gap={gap:.3f} "
                f"precision={test_precision:.3f} recall={test_recall:.3f}"
            )
        else:
            logger.warning(
                f"ML GATE FAILED: {self.MODEL_NAME} v{version} "
                f"train={train_acc:.3f} test={test_acc:.3f} gap={gap:.3f} "
                f"(need test>{self.DEPLOY_GATE['min_test_acc']}, gap<{self.DEPLOY_GATE['max_gap']})"
            )

        self.feature_names = available_features

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "gap": gap,
            "deployed": gate_passed,
            "model_version": version,
            "precision": test_precision,
            "recall": test_recall,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def _compute_binary_labels(self, daily: pd.DataFrame) -> pd.Series:
        """
        CE binary labels: 1 if next_day_return > +0.2% (bullish day).
        """
        next_open = daily["open"].shift(-1)
        next_close = daily["close"].shift(-1)
        next_return = (next_close - next_open) / next_open
        return (next_return > self.MOVE_THRESHOLD).astype(int)

    PE_DROP_WINDOW = 6  # 6 candles = 30 min

    def _compute_pe_fast_drop_labels(
        self, candles_5min: pd.DataFrame, daily: pd.DataFrame,
    ) -> pd.Series:
        """
        PE V2 label: 1 if NEXT day has a drop ≥0.25% in any 6 consecutive 5-min candles (30 min).

        Captures profitable PE moves — not just fast crashes but sustained selling pressure.
        Uses rolling 6-bar window on 5-min data for the NEXT trading day.
        """
        df = candles_5min.copy()
        df["date"] = df["datetime"].dt.date.astype(str)
        window = self.PE_DROP_WINDOW

        # For each day, check if any N consecutive 5-min bars drop ≥ PE_FAST_DROP
        fast_drop_days = set()
        for dt, group in df.groupby("date"):
            if len(group) < window:
                continue
            closes = group["close"].values
            # Rolling N-bar return: (close[i+N-1] - close[i]) / close[i]
            for i in range(len(closes) - window + 1):
                if closes[i] > 0:
                    ret = (closes[i + window - 1] - closes[i]) / closes[i]
                    if ret <= -self.PE_FAST_DROP:
                        fast_drop_days.add(dt)
                        break

        # Label: 1 if NEXT day is a fast-drop day
        dates = daily["date"].values
        labels = []
        for i in range(len(dates)):
            if i + 1 < len(dates):
                next_date = dates[i + 1]
                labels.append(1 if next_date in fast_drop_days else 0)
            else:
                labels.append(np.nan)

        return pd.Series(labels, index=daily.index)

    CE_RISE_WINDOW = 6  # 6 candles = 30 min

    def _compute_ce_fast_rise_labels(
        self, candles_5min: pd.DataFrame, daily: pd.DataFrame,
    ) -> pd.Series:
        """
        CE V2 label: 1 if NEXT day has a rise ≥0.25% in any 6 consecutive 5-min candles (30 min).

        Captures profitable CE moves — sustained buying pressure within a 30-min window.
        Uses rolling 6-bar window on 5-min data for the NEXT trading day.
        """
        df = candles_5min.copy()
        df["date"] = df["datetime"].dt.date.astype(str)
        window = self.CE_RISE_WINDOW

        # For each day, check if any N consecutive 5-min bars rise ≥ CE_FAST_RISE
        fast_rise_days = set()
        for dt, group in df.groupby("date"):
            if len(group) < window:
                continue
            closes = group["close"].values
            for i in range(len(closes) - window + 1):
                if closes[i] > 0:
                    ret = (closes[i + window - 1] - closes[i]) / closes[i]
                    if ret >= self.CE_FAST_RISE:
                        fast_rise_days.add(dt)
                        break

        # Label: 1 if NEXT day is a fast-rise day
        dates = daily["date"].values
        labels = []
        for i in range(len(dates)):
            if i + 1 < len(dates):
                next_date = dates[i + 1]
                labels.append(1 if next_date in fast_rise_days else 0)
            else:
                labels.append(np.nan)

        return pd.Series(labels, index=daily.index)

    def _check_deploy_gate(self, train_acc: float, test_acc: float, precision: float = 1.0) -> bool:
        """Deploy gate: test_acc > 52%, gap < 15%, PE precision > 50%."""
        gap = train_acc - test_acc
        min_precision = self.DEPLOY_GATE.get("min_precision", 0.0)
        return (
            test_acc > self.DEPLOY_GATE["min_test_acc"]
            and gap < self.DEPLOY_GATE["max_gap"]
            and precision >= min_precision
        )

    def _get_xgb_params(self, scale_pos_weight: float = 1.0) -> dict:
        """XGBoost params for binary direction model. PE uses shallower trees + more rounds."""
        if self.side == "pe":
            return {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "max_depth": 3,
                "learning_rate": 0.03,
                "n_estimators": 300,
                "min_child_weight": 25,
                "subsample": 0.7,
                "colsample_bytree": 0.65,
                "reg_alpha": 2.5,
                "reg_lambda": 2.5,
                "gamma": 0.4,
                "scale_pos_weight": round(scale_pos_weight, 2),
                "tree_method": "hist",
                "verbosity": 0,
            }
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 200,
            "min_child_weight": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.65,
            "reg_alpha": 2.0,
            "reg_lambda": 2.0,
            "gamma": 0.3,
            "scale_pos_weight": round(scale_pos_weight, 2),
            "tree_method": "hist",
            "verbosity": 0,
        }

    def predict(self, features: dict) -> dict:
        """
        Predict probability that this side's condition is true.

        Returns: {prob: float, prediction: 0/1}
        """
        if self.model is None or self.scaler is None:
            return {"prob": 0.5, "prediction": 0}

        feature_vals = [features.get(f, 0.0) for f in self.feature_names]
        X = pd.DataFrame([feature_vals], columns=self.feature_names)
        X_scaled = self.scaler.transform(X)

        prob = float(self.model.predict_proba(X_scaled)[0][1])
        return {"prob": prob, "prediction": 1 if prob > 0.5 else 0}

    def load_deployed_model(self) -> bool:
        """Load the currently deployed model from DB + disk."""
        record = self.store.get_deployed_model(self.MODEL_NAME)
        if not record:
            logger.debug(f"No deployed {self.MODEL_NAME} model found")
            return False

        model_path = Path(record["model_path"])
        scaler_path = Path(record["scaler_path"]) if record.get("scaler_path") else None

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            return False

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            if scaler_path and scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)

            self.model_version = record["model_version"]
            if record.get("feature_list"):
                self.feature_names = json.loads(record["feature_list"])

            logger.info(
                f"ML LOADED: {self.MODEL_NAME} v{self.model_version} "
                f"test_acc={record['test_accuracy']:.1%}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load {self.MODEL_NAME}: {e}")
            return False


def predict_direction_v2(
    pe_trainer: BinaryDirectionTrainer,
    ce_trainer: BinaryDirectionTrainer,
    features: dict,
) -> dict:
    """
    Combine PE and CE binary model predictions into a single direction call.

    Logic:
      pe_prob > 0.55 and pe_prob > ce_prob → PE direction
      ce_prob > 0.55 and ce_prob > pe_prob → CE direction
      Otherwise → FLAT (no strong signal)

    Returns: {direction, pe_prob, ce_prob, confidence, source}
    """
    pe_result = pe_trainer.predict(features)
    ce_result = ce_trainer.predict(features)

    pe_prob = pe_result["prob"]
    ce_prob = ce_result["prob"]

    if pe_prob > 0.55 and pe_prob > ce_prob:
        direction = "PE"
        confidence = pe_prob
    elif ce_prob > 0.55 and ce_prob > pe_prob:
        direction = "CE"
        confidence = ce_prob
    else:
        direction = "FLAT"
        confidence = 0.5

    return {
        "direction": direction,
        "pe_prob": pe_prob,
        "ce_prob": ce_prob,
        "confidence": confidence,
        "source": "v2_binary",
    }


class QualityModelTrainer:
    """Stage 2: Train trade quality prediction model."""

    MODEL_NAME = "quality_v1"
    MODEL_DIR = Path("data/models")
    DEPLOY_GATE = {"min_test_acc": 0.58, "max_gap": 0.18, "min_rows": 30}

    QUALITY_FEATURES = [
        "score_diff", "conviction", "vix_at_entry", "rsi_at_entry",
        "adx_at_entry", "pcr_at_entry", "ml_prob_ce", "ml_prob_pe",
        "trigger_count", "regime_encoded", "direction_encoded", "days_to_expiry",
    ]

    def __init__(self, store: DataStore):
        self.store = store
        self.model = None
        self.scaler = None
        self.model_version: int = 0

    def train(self) -> dict:
        """Train quality model from labeled trades."""
        if not XGB_AVAILABLE:
            return {"error": "xgboost not installed", "deployed": False}

        labels_df = self.store.get_ml_trade_labels(limit=500)
        if labels_df.empty or len(labels_df) < self.DEPLOY_GATE["min_rows"]:
            return {
                "error": f"Insufficient labels: {len(labels_df)}/{self.DEPLOY_GATE['min_rows']}",
                "deployed": False,
            }

        # Extract features
        X_rows = []
        y_vals = []
        for _, row in labels_df.iterrows():
            feat = {}
            for col in self.QUALITY_FEATURES:
                if col in labels_df.columns:
                    feat[col] = float(row[col]) if pd.notna(row[col]) else 0.0
                elif col == "regime_encoded":
                    regime_map = {"TRENDING": 0, "RANGEBOUND": 1, "VOLATILE": 2, "ELEVATED": 3}
                    feat[col] = regime_map.get(row.get("regime", ""), 1)
                elif col == "direction_encoded":
                    feat[col] = 1.0 if row.get("direction") == "CE" else 0.0
                elif col == "days_to_expiry":
                    feat[col] = 3.0
                else:
                    feat[col] = 0.0
            X_rows.append(feat)
            y_vals.append(int(row["label"]))

        X = pd.DataFrame(X_rows)
        y = pd.Series(y_vals)

        # Walk-forward split (last 20%, min 6 trades)
        n_test = max(6, len(X) // 5)
        split_idx = len(X) - n_test
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"ML TRAIN quality: {len(X_train)} train, {len(X_test)} test")

        # Scale
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train
        params = self._get_xgb_params()
        n_estimators = params.pop("n_estimators", 50)

        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            **params,
        )
        self.model.fit(X_train_scaled, y_train, verbose=False)

        # Evaluate
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)

        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        gap = train_acc - test_acc

        gate_passed = self._check_deploy_gate(train_acc, test_acc, len(labels_df))

        # Save (paths filled after we get the version)
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        version = self.store.save_ml_model_record({
            "model_name": self.MODEL_NAME,
            "model_type": "xgboost",
            "stage": "quality",
            "train_date": date.today().isoformat(),
            "train_samples": len(X_train),
            "n_features": len(self.QUALITY_FEATURES),
            "train_accuracy": train_acc,
            "test_accuracy": test_acc,
            "train_test_gap": gap,
            "deployed": 1 if gate_passed else 0,
            "deploy_gate_passed": 1 if gate_passed else 0,
            "model_path": "",
            "scaler_path": "",
            "feature_list": self.QUALITY_FEATURES,
            "hyperparams": self._get_xgb_params(),
        })

        self.model_version = version

        if gate_passed:
            model_path = self.MODEL_DIR / f"{self.MODEL_NAME}_v{version}.pkl"
            scaler_path = self.MODEL_DIR / f"{self.MODEL_NAME}_scaler_v{version}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)
            with open(scaler_path, "wb") as f:
                pickle.dump(self.scaler, f)

            self.store.update_ml_model_paths(
                self.MODEL_NAME, version, str(model_path), str(scaler_path)
            )
            self.store.set_model_deployed(self.MODEL_NAME, version)
            logger.info(f"ML DEPLOYED: {self.MODEL_NAME} v{version} test={test_acc:.3f}")

        return {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "gap": gap,
            "deployed": gate_passed,
            "model_version": version,
            "n_train": len(X_train),
            "n_test": len(X_test),
        }

    def predict(self, trade_features: dict) -> dict:
        """Predict win probability for a trade setup."""
        if self.model is None or self.scaler is None:
            return {"win_prob": 0.5, "quality_class": "UNKNOWN"}

        feature_vals = [trade_features.get(f, 0.0) for f in self.QUALITY_FEATURES]
        X = pd.DataFrame([feature_vals], columns=self.QUALITY_FEATURES)
        X_scaled = self.scaler.transform(X)

        win_prob = float(self.model.predict_proba(X_scaled)[0][1])
        quality_class = "HIGH" if win_prob >= 0.55 else "LOW"

        return {"win_prob": win_prob, "quality_class": quality_class}

    def _check_deploy_gate(self, train_acc: float, test_acc: float, n_rows: int) -> bool:
        """Gate: test_acc > 58%, gap < 18%, n_rows >= 30."""
        gap = train_acc - test_acc
        return (
            test_acc > self.DEPLOY_GATE["min_test_acc"]
            and gap < self.DEPLOY_GATE["max_gap"]
            and n_rows >= self.DEPLOY_GATE["min_rows"]
        )

    def _get_xgb_params(self) -> dict:
        """XGBoost params for binary quality model."""
        return {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "learning_rate": 0.1,
            "n_estimators": 50,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 2.0,
            "reg_lambda": 2.0,
            "tree_method": "hist",
            "verbosity": 0,
        }

    def load_deployed_model(self) -> bool:
        """Load the currently deployed quality model."""
        record = self.store.get_deployed_model(self.MODEL_NAME)
        if not record:
            return False

        model_path = Path(record["model_path"])
        scaler_path = Path(record["scaler_path"]) if record.get("scaler_path") else None

        if not model_path.exists():
            return False

        try:
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
            if scaler_path and scaler_path.exists():
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
            self.model_version = record["model_version"]
            logger.info(f"ML LOADED: {self.MODEL_NAME} v{self.model_version}")
            return True
        except Exception as e:
            logger.error(f"Failed to load {self.MODEL_NAME}: {e}")
            return False


class DriftDetector:
    """Monitor prediction accuracy drift over rolling window."""

    DRIFT_THRESHOLD = 0.10  # 10% accuracy drop

    def __init__(self, store: DataStore):
        self.store = store

    MIN_PREDICTIONS_FOR_DRIFT = 10  # Need 10+ predictions since deploy

    def check_drift(self, model_name: str, window_days: int = 20) -> dict:
        """
        Compare recent prediction accuracy vs training accuracy.

        Only uses predictions made AFTER the deployed model's train_date
        (i.e., predictions from the current deployment, not old versions).

        Returns: {drifted, baseline_acc, recent_acc, drop, n_predictions}
        """
        deployed = self.store.get_deployed_model(model_name)
        if not deployed:
            return {"drifted": False, "baseline_acc": 0, "recent_acc": 0, "drop": 0, "n_predictions": 0}

        baseline_acc = deployed.get("test_accuracy", 0)
        deploy_date = deployed.get("train_date", "")

        # Only fetch predictions made AFTER deploy date
        predictions = self.store.get_ml_predictions(
            model_name, from_date=deploy_date, limit=window_days * 2,
        )
        if predictions.empty:
            return {"drifted": False, "baseline_acc": baseline_acc, "recent_acc": 0, "drop": 0, "n_predictions": 0}

        # Only use labeled predictions
        labeled = predictions[predictions["actual_class"].notna()]
        if len(labeled) < self.MIN_PREDICTIONS_FOR_DRIFT:
            logger.debug(
                f"DRIFT_SKIP: only {len(labeled)} predictions since deploy "
                f"(need {self.MIN_PREDICTIONS_FOR_DRIFT}+)"
            )
            return {"drifted": False, "baseline_acc": baseline_acc, "recent_acc": 0, "drop": 0, "n_predictions": len(labeled)}

        recent_acc = labeled["correct"].mean()
        drop = baseline_acc - recent_acc

        return {
            "drifted": drop > self.DRIFT_THRESHOLD,
            "baseline_acc": baseline_acc,
            "recent_acc": float(recent_acc),
            "drop": float(drop),
            "n_predictions": len(labeled),
        }

    def should_retrain(self, model_name: str) -> bool:
        """Returns True if drift detected or no deployed model exists."""
        deployed = self.store.get_deployed_model(model_name)
        if not deployed:
            return True
        result = self.check_drift(model_name)
        return result.get("drifted", False)
