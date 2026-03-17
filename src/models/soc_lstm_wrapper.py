"""
soc_lstm_wrapper.py — LSTM State-of-Charge Model Wrapper

Wraps a trained LSTM (from SoC_Estimation_LSTM.ipynb or any Keras/PyTorch
LSTM) into the exact same BaseBMSModel interface used by SOHRegressor and
FaultClassifier, so it plugs into EnsembleBMSPredictor with zero changes
to the existing monitoring loop.

What this file does
───────────────────
1. SoCSequenceBuilder      — turns a flat DataFrame into sliding-window
                             sequences for LSTM input  (N, seq_len, features)
2. SoCLSTMWrapper          — wraps a Keras .h5 / .keras model
3. SoCLSTMFallbackWrapper  — pure sklearn MLP approximation (no TensorFlow)
4. train_soc_lstm()        — end-to-end training; auto-selects backend

Typical notebook architecture (what this wraps)
───────────────────────────────────────────────
Input  : (batch, seq_len=10, features=3)   [Voltage_V, Current_A, Temp_C]
LSTM(64, return_sequences=True) → Dropout(0.2)
LSTM(32)                        → Dropout(0.2)
Dense(1, activation='sigmoid')             [SoC ∈ 0..1]

Quick-start
───────────
  # A) Load model saved by the notebook (.h5 file):
  from src.models.soc_lstm_wrapper import SoCLSTMWrapper
  wrapper = SoCLSTMWrapper.from_keras_file("models/soc_model.h5",
                                           scaler_path="models/scaler.pkl")
  wrapper.save()   # → models/soc_lstm.joblib

  # B) Train from scratch:
  from src.models.soc_lstm_wrapper import train_soc_lstm
  from src.data.battery_dataset_adapter import BatteryDatasetAdapter
  df = BatteryDatasetAdapter.load_and_normalize("data/battery_data.csv")
  result = train_soc_lstm(df)

  # C) Add to ensemble (one line):
  from src.models.soc_lstm_wrapper import SoCLSTMWrapper
  ensemble.soc_lstm_model = SoCLSTMWrapper.load()
"""

from __future__ import annotations

import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

BASE_DIR   = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SOC_INPUT_FEATURES = ["Voltage_V", "Current_A", "Temperature_C"]
SOC_TARGET         = "SoC"
DEFAULT_SEQ_LEN    = 10


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Sequence builder
# ─────────────────────────────────────────────────────────────────────────────

class SoCSequenceBuilder:
    """
    Converts flat battery-telemetry rows into overlapping 3-D windows
    ready for LSTM input.

        DataFrame rows  →  X: (N, seq_len, n_features)
                           y: (N,)   SoC at end of each window
    """

    def __init__(
        self,
        seq_len:      int       = DEFAULT_SEQ_LEN,
        feature_cols: List[str] = None,
        target_col:   str       = SOC_TARGET,
        step:         int       = 1,
    ):
        self.seq_len      = seq_len
        self.feature_cols = feature_cols or SOC_INPUT_FEATURES
        self.target_col   = target_col
        self.step         = step
        self._feature_min: Optional[np.ndarray] = None
        self._feature_max: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "SoCSequenceBuilder":
        X_raw = df[self.feature_cols].values.astype(np.float32)
        self._feature_min = X_raw.min(axis=0)
        self._feature_max = X_raw.max(axis=0)
        self._feature_max = np.where(
            self._feature_max == self._feature_min,
            self._feature_min + 1.0,
            self._feature_max,
        )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if not self._fitted:
            raise RuntimeError("Call .fit() on training data before .transform().")
        missing = [c for c in self.feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns {missing}. "
                f"Run BatteryDatasetAdapter.load_and_normalize() first."
            )
        X_raw    = df[self.feature_cols].values.astype(np.float32)
        y_raw    = df[self.target_col].values.astype(np.float32) \
                   if self.target_col in df.columns else None
        X_scaled = (X_raw - self._feature_min) / (self._feature_max - self._feature_min)
        n        = len(X_scaled)
        indices  = range(self.seq_len - 1, n, self.step)
        X_seq    = np.stack([X_scaled[i - self.seq_len + 1: i + 1] for i in indices])
        y_seq    = np.array([y_raw[i] for i in indices]) if y_raw is not None else None
        return X_seq, y_seq

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        return self.fit(df).transform(df)

    def transform_single(self, window_df: pd.DataFrame) -> np.ndarray:
        """Single window → (1, seq_len, n_features) for real-time inference."""
        if not self._fitted:
            raise RuntimeError("Call .fit() first.")
        X_raw    = window_df[self.feature_cols].tail(self.seq_len).values.astype(np.float32)
        X_scaled = (X_raw - self._feature_min) / (self._feature_max - self._feature_min)
        return X_scaled[np.newaxis, :, :]


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Keras LSTM wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SoCLSTMWrapper:
    """
    Wraps a Keras LSTM model in the BaseBMSModel-compatible interface.
    Supports .h5, .keras, and SavedModel formats from the notebook.
    """

    name = "soc_lstm"
    task = "regression"

    def __init__(
        self,
        seq_len:       int       = DEFAULT_SEQ_LEN,
        feature_cols:  List[str] = None,
        lstm_units:    List[int] = None,
        dropout:       float     = 0.2,
        epochs:        int       = 30,
        batch_size:    int       = 64,
        learning_rate: float     = 0.001,
    ):
        self.seq_len       = seq_len
        self.feature_cols  = feature_cols or SOC_INPUT_FEATURES
        self.lstm_units    = lstm_units or [64, 32]
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.seq_builder_: Optional[SoCSequenceBuilder] = None
        self._keras_model  = None
        self._weights_     = None
        self._history_     = None

    # ── Build ─────────────────────────────────────────────────────────────

    def _build_keras_model(self):
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError(
                "TensorFlow is required. Install: pip install tensorflow\n"
                "Or use SoCLSTMFallbackWrapper for a no-TF approximation."
            )
        n_features = len(self.feature_cols)
        model = keras.Sequential(name="SoC_LSTM")
        for i, units in enumerate(self.lstm_units):
            ret_seq = (i < len(self.lstm_units) - 1)
            if i == 0:
                model.add(keras.layers.LSTM(
                    units, return_sequences=ret_seq,
                    input_shape=(self.seq_len, n_features), name=f"lstm_{i}"
                ))
            else:
                model.add(keras.layers.LSTM(units, return_sequences=ret_seq, name=f"lstm_{i}"))
            model.add(keras.layers.Dropout(self.dropout, name=f"dropout_{i}"))
        model.add(keras.layers.Dense(1, activation="sigmoid", name="soc_output"))
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse", metrics=["mae"],
        )
        return model

    # ── Train ─────────────────────────────────────────────────────────────

    def train(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame] = None,
              verbose: int = 0) -> "SoCLSTMWrapper":
        self.seq_builder_ = SoCSequenceBuilder(
            seq_len=self.seq_len, feature_cols=self.feature_cols
        )
        X_train, y_train = self.seq_builder_.fit_transform(df_train)
        logger.info(
            f"SoCLSTMWrapper training: {X_train.shape} sequences, "
            f"SoC range=[{y_train.min():.3f}, {y_train.max():.3f}]"
        )
        val_data = None
        if df_val is not None:
            X_val, y_val = self.seq_builder_.transform(df_val)
            val_data = (X_val, y_val)

        self._keras_model = self._build_keras_model()
        try:
            from tensorflow import keras
            cbs = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss" if val_data else "loss",
                    patience=5, restore_best_weights=True, verbose=0),
                keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss" if val_data else "loss",
                    factor=0.5, patience=3, verbose=0),
            ]
            self._history_ = self._keras_model.fit(
                X_train, y_train,
                validation_data=val_data,
                epochs=self.epochs, batch_size=self.batch_size,
                callbacks=cbs, verbose=verbose,
            )
        except Exception as e:
            logger.warning(f"Keras training error: {e}")

        self._weights_ = [w.numpy() for w in self._keras_model.weights]
        final_loss = self._history_.history["loss"][-1] if self._history_ else "N/A"
        logger.success(f"SoCLSTMWrapper trained — final loss={final_loss}")
        return self

    # ── Predict ───────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._ensure_ready()
        X_seq, _ = self.seq_builder_.transform(X)
        return np.clip(self._keras_model.predict(X_seq, verbose=0).flatten(), 0.0, 1.0)

    def predict_realtime(self, window_df: pd.DataFrame) -> float:
        """Single-window real-time edge inference. Returns scalar SoC ∈ [0,1]."""
        self._ensure_ready()
        X = self.seq_builder_.transform_single(window_df)
        return float(np.clip(self._keras_model.predict(X, verbose=0)[0, 0], 0.0, 1.0))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        preds = self.predict(X)
        y_al  = y.values[-len(preds):]
        return {
            "mae":       round(float(mean_absolute_error(y_al, preds)), 4),
            "rmse":      round(float(np.sqrt(mean_squared_error(y_al, preds))), 4),
            "r2":        round(float(r2_score(y_al, preds)), 4),
            "mean_pred": round(float(preds.mean()), 4),
            "n_samples": len(preds),
        }

    # ── Load from notebook file ───────────────────────────────────────────

    @classmethod
    def from_keras_file(
        cls,
        keras_path:   str | Path,
        seq_len:      int       = DEFAULT_SEQ_LEN,
        feature_cols: List[str] = None,
        scaler_path:  Optional[str | Path] = None,
    ) -> "SoCLSTMWrapper":
        """
        ★ MAIN ENTRY POINT for loading YOUR notebook model ★

        Load a Keras model saved by SoC_Estimation_LSTM.ipynb:
            model.save("soc_model.h5")       →  keras_path="soc_model.h5"
            model.save("soc_model.keras")    →  keras_path="soc_model.keras"

        scaler_path: path to the sklearn MinMaxScaler saved by the notebook,
                     e.g. joblib.dump(scaler, "scaler.pkl")
                     If not provided, you must call
                     wrapper.seq_builder_.fit(df_train) manually.
        """
        try:
            from tensorflow import keras as _k
        except ImportError:
            raise ImportError("pip install tensorflow")

        keras_path = Path(keras_path)
        if not keras_path.exists():
            raise FileNotFoundError(f"Not found: {keras_path}")

        w = cls(seq_len=seq_len, feature_cols=feature_cols)
        w._keras_model = _k.models.load_model(str(keras_path))
        w._weights_    = [ww.numpy() for ww in w._keras_model.weights]
        logger.info(f"Loaded Keras model from {keras_path}")

        w.seq_builder_ = SoCSequenceBuilder(
            seq_len=seq_len,
            feature_cols=feature_cols or SOC_INPUT_FEATURES,
        )
        if scaler_path and Path(scaler_path).exists():
            nb_scaler = joblib.load(scaler_path)
            nf = len(SOC_INPUT_FEATURES)
            w.seq_builder_._feature_min = np.array(nb_scaler.data_min_[:nf])
            w.seq_builder_._feature_max = np.array(nb_scaler.data_max_[:nf])
            w.seq_builder_._fitted = True
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            logger.warning(
                "No scaler loaded. Call wrapper.seq_builder_.fit(df_train) "
                "with the same training data used in the notebook."
            )
        return w

    # ── Persist ───────────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"{self.name}.joblib")
        joblib.dump({
            "config": {
                "seq_len": self.seq_len, "feature_cols": self.feature_cols,
                "lstm_units": self.lstm_units, "dropout": self.dropout,
                "learning_rate": self.learning_rate,
            },
            "weights":     self._weights_,
            "seq_builder": self.seq_builder_,
        }, path)
        logger.info(f"Saved SoCLSTMWrapper → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SoCLSTMWrapper":
        path = path or (MODELS_DIR / "soc_lstm.joblib")
        if not path.exists():
            raise FileNotFoundError(f"No saved model at {path}")
        obj = joblib.load(path)
        w   = cls(**obj["config"])
        w.seq_builder_ = obj["seq_builder"]
        w._weights_    = obj["weights"]
        try:
            import tensorflow as tf
            w._keras_model = w._build_keras_model()
            for lw, sw in zip(w._keras_model.weights, w._weights_):
                lw.assign(tf.constant(sw))
            logger.info(f"Loaded SoCLSTMWrapper from {path}")
        except ImportError:
            logger.warning("TF not available — use SoCLSTMFallbackWrapper.load()")
        return w

    def _ensure_ready(self):
        if self._keras_model is None:
            raise RuntimeError("Model not loaded. Call .train() or .from_keras_file().")
        if self.seq_builder_ is None or not self.seq_builder_._fitted:
            raise RuntimeError("Sequence builder not fitted. Call seq_builder_.fit(df_train).")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sklearn fallback (no TensorFlow needed)
# ─────────────────────────────────────────────────────────────────────────────

class SoCLSTMFallbackWrapper:
    """
    A sklearn MLP approximation of the LSTM — same interface, no TensorFlow.
    Automatically used when TF is not installed.
    """

    name = "soc_lstm_fallback"
    task = "regression"

    def __init__(
        self,
        seq_len:            int   = DEFAULT_SEQ_LEN,
        feature_cols:       List[str] = None,
        hidden_layer_sizes: tuple = (128, 64, 32),
        max_iter:           int   = 200,
    ):
        self.seq_len            = seq_len
        self.feature_cols       = feature_cols or SOC_INPUT_FEATURES
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter           = max_iter
        self.seq_builder_: Optional[SoCSequenceBuilder] = None
        self._mlp_ = None

    def train(self, df: pd.DataFrame, **_) -> "SoCLSTMFallbackWrapper":
        from sklearn.neural_network import MLPRegressor
        self.seq_builder_ = SoCSequenceBuilder(
            seq_len=self.seq_len, feature_cols=self.feature_cols
        )
        X_seq, y_seq = self.seq_builder_.fit_transform(df)
        X_flat = X_seq.reshape(len(X_seq), -1)
        self._mlp_ = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter, random_state=42,
            early_stopping=True, n_iter_no_change=10,
        )
        self._mlp_.fit(X_flat, y_seq)
        logger.info(f"SoCLSTMFallbackWrapper trained — iters={self._mlp_.n_iter_}")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_seq, _ = self.seq_builder_.transform(X)
        return np.clip(self._mlp_.predict(X_seq.reshape(len(X_seq), -1)), 0.0, 1.0)

    def predict_realtime(self, window_df: pd.DataFrame) -> float:
        X = self.seq_builder_.transform_single(window_df)
        return float(np.clip(self._mlp_.predict(X.reshape(1, -1))[0], 0.0, 1.0))

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        preds = self.predict(X)
        y_al  = y.values[-len(preds):]
        return {
            "mae":       round(float(mean_absolute_error(y_al, preds)), 4),
            "rmse":      round(float(np.sqrt(mean_squared_error(y_al, preds))), 4),
            "r2":        round(float(r2_score(y_al, preds)), 4),
            "mean_pred": round(float(preds.mean()), 4),
        }

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / f"{self.name}.joblib")
        joblib.dump(self, path)
        logger.info(f"Saved SoCLSTMFallbackWrapper → {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "SoCLSTMFallbackWrapper":
        path = path or (MODELS_DIR / "soc_lstm_fallback.joblib")
        return joblib.load(path)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Auto-selecting train function
# ─────────────────────────────────────────────────────────────────────────────

def train_soc_lstm(
    df_train:       pd.DataFrame,
    df_val:         Optional[pd.DataFrame] = None,
    seq_len:        int        = DEFAULT_SEQ_LEN,
    lstm_units:     List[int]  = None,
    epochs:         int        = 30,
    save:           bool       = True,
    force_fallback: bool       = False,
) -> Dict:
    """
    Train a SoC LSTM — auto-detects TensorFlow and picks the best backend.

    Args:
        df_train       : Training DataFrame with Voltage_V, Current_A,
                         Temperature_C, SoC  (use BatteryDatasetAdapter first)
        df_val         : Optional validation split
        seq_len        : Look-back window (default 10)
        lstm_units     : LSTM layer sizes (default [64, 32])
        epochs         : Max epochs — early stopping applies
        save           : Save model to models/ directory
        force_fallback : Skip TF even if available (for testing)

    Returns: dict  {model, metrics, model_type, saved_path}
    """
    tf_ok = False
    if not force_fallback:
        try:
            import tensorflow  # noqa
            tf_ok = True
        except ImportError:
            pass

    if tf_ok:
        logger.info("Using Keras LSTM backend")
        model      = SoCLSTMWrapper(seq_len=seq_len, lstm_units=lstm_units or [64, 32], epochs=epochs)
        model_type = "keras_lstm"
        model.train(df_train, df_val=df_val)
    else:
        logger.info("TF not found — using sklearn MLP fallback")
        model      = SoCLSTMFallbackWrapper(seq_len=seq_len)
        model_type = "sklearn_mlp_fallback"
        model.train(df_train)

    eval_df    = df_val if df_val is not None else df_train
    metrics    = model.evaluate(eval_df, eval_df[SOC_TARGET])
    saved_path = model.save() if save else None

    logger.info(f"SoC model [{model_type}] → {metrics}")
    return {"model": model, "metrics": metrics, "model_type": model_type, "saved_path": saved_path}
