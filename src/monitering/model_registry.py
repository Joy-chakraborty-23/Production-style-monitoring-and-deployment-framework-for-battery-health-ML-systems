"""
model_registry.py — Multi-Model Registry

Manages a fleet of production ML models in a single database.
Supports SQLite locally (zero config) and PostgreSQL in production
via the same interface — just change the connection URL.

Schema:
  models        — registered model metadata + per-model thresholds
  batch_metrics — time-series AUC/F1/drift per model per batch
  retrain_log   — every retrain event per model
  drift_details — per-feature drift scores per model per batch

Usage:
    # Local SQLite (development / demo)
    registry = ModelRegistry()                        # uses models/registry.db
    registry = ModelRegistry("sqlite:///my.db")

    # PostgreSQL (production)
    registry = ModelRegistry("postgresql://user:pass@host:5432/mlmonitor")

    # Register a model
    model_id = registry.register_model(
        name="churn_xgboost",
        version="1.0.0",
        description="Customer churn prediction — XGBoost pipeline",
        pipeline_path="models/churn_pipeline.joblib",
        thresholds={"auc_warning": 0.80, "auc_critical": 0.70, "psi_critical": 0.20},
        tags={"domain": "retention", "team": "ds-platform"},
    )

    # Log a batch
    registry.log_batch_metrics(model_id, batch_id=5, auc_roc=0.83, ...)

    # Fleet health summary
    df = registry.get_fleet_health()
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

from src.config import MODELS_DIR


# ── Default thresholds applied when a model is registered without custom ones ─
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "auc_warning":   0.80,
    "auc_critical":  0.70,
    "f1_warning":    0.75,
    "psi_warning":   0.10,
    "psi_critical":  0.20,
}

# ── Status codes ──────────────────────────────────────────────────────────────
STATUS_HEALTHY  = "healthy"
STATUS_WARNING  = "warning"
STATUS_CRITICAL = "critical"
STATUS_INACTIVE = "inactive"
STATUS_UNKNOWN  = "unknown"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class RegisteredModel:
    model_id:     int
    name:         str
    version:      str
    description:  str
    pipeline_path: str
    thresholds:   Dict[str, float]
    tags:         Dict[str, str]
    registered_at: str
    status:       str = STATUS_INACTIVE
    last_batch_id: Optional[int] = None
    latest_auc:   Optional[float] = None
    latest_drift_severity: str = "none"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FleetHealthRow:
    model_id:     int
    name:         str
    version:      str
    status:       str
    latest_auc:   Optional[float]
    latest_f1:    Optional[float]
    drift_severity: str
    last_batch_id: Optional[int]
    last_updated: Optional[str]
    n_batches:    int
    n_retrains:   int
    roi_monthly_loss: Optional[float]
    description:  str


# ── Registry class ────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Central registry for all production ML models.

    Wraps either SQLite (local, zero-config) or PostgreSQL (production).
    The public API is identical for both — swap the db_url to switch.

    All writes are transactional.  All reads return pandas DataFrames
    so results plug directly into the monitoring loop and dashboard.
    """

    def __init__(self, db_url: Optional[str] = None):
        """
        Args:
            db_url: Connection string.
                    None → SQLite at models/registry.db  (default)
                    "sqlite:///path/to/file.db"
                    "postgresql://user:pass@host:5432/dbname"
        """
        if db_url is None:
            db_path = MODELS_DIR / "registry.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_url = f"sqlite:///{db_path}"

        self.db_url  = db_url
        self._is_pg  = db_url.startswith("postgresql")
        self._sqlite_path = db_url.replace("sqlite:///", "") if not self._is_pg else None

        self._init_schema()
        logger.info(f"ModelRegistry initialised — {db_url}")

    # ── Internal DB helpers ───────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        """Yield a live connection, auto-commit or rollback."""
        if self._is_pg:
            # Lazy import — only required for PostgreSQL deployments
            try:
                import psycopg2
                conn = psycopg2.connect(self.db_url.replace("postgresql://", ""))
            except ImportError:
                raise ImportError(
                    "psycopg2 is required for PostgreSQL. "
                    "Install it: pip install psycopg2-binary"
                )
        else:
            conn = sqlite3.connect(self._sqlite_path)
            conn.row_factory = sqlite3.Row

        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _placeholder(self) -> str:
        """Return the correct parameter placeholder for the current DB."""
        return "%s" if self._is_pg else "?"

    def _execute(self, sql: str, params: tuple = ()) -> None:
        """Execute a non-returning statement."""
        ph = self._placeholder()
        sql = sql.replace("?", ph)
        with self._conn() as conn:
            conn.cursor().execute(sql, params)

    def _fetchall(self, sql: str, params: tuple = ()) -> List[dict]:
        """Execute a SELECT and return list of dicts."""
        ph = self._placeholder()
        sql = sql.replace("?", ph)
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            if self._is_pg:
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, row)) for row in cur.fetchall()]
            else:
                return [dict(row) for row in cur.fetchall()]

    # ── Schema creation ───────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create tables if they do not already exist."""
        stmts = [
            """
            CREATE TABLE IF NOT EXISTS models (
                model_id      INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                version       TEXT NOT NULL DEFAULT '1.0.0',
                description   TEXT DEFAULT '',
                pipeline_path TEXT DEFAULT '',
                thresholds    TEXT DEFAULT '{}',
                tags          TEXT DEFAULT '{}',
                registered_at TEXT NOT NULL,
                status        TEXT DEFAULT 'inactive',
                UNIQUE(name, version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS batch_metrics (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id       INTEGER NOT NULL REFERENCES models(model_id),
                batch_id       INTEGER NOT NULL,
                recorded_at    TEXT NOT NULL,
                auc_roc        REAL,
                f1_score       REAL,
                precision_val  REAL,
                recall_val     REAL,
                brier_score    REAL,
                drift_score    REAL,
                drift_severity TEXT DEFAULT 'none',
                n_drifted_feats INTEGER DEFAULT 0,
                n_samples      INTEGER DEFAULT 0,
                retrain_triggered INTEGER DEFAULT 0,
                roi_monthly_loss  REAL DEFAULT 0,
                roi_missed_churners INTEGER DEFAULT 0,
                extra_json     TEXT DEFAULT '{}'
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS retrain_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id        INTEGER NOT NULL REFERENCES models(model_id),
                triggered_at    TEXT NOT NULL,
                trigger_batch   INTEGER,
                trigger_reason  TEXT DEFAULT '',
                old_auc         REAL,
                new_auc         REAL,
                promoted        INTEGER DEFAULT 0,
                promotion_reason TEXT DEFAULT '',
                mlflow_run_id   TEXT DEFAULT ''
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS drift_details (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id   INTEGER NOT NULL REFERENCES models(model_id),
                batch_id   INTEGER NOT NULL,
                feature    TEXT NOT NULL,
                method     TEXT DEFAULT '',
                score      REAL,
                p_value    REAL,
                severity   TEXT DEFAULT 'none',
                is_drifted INTEGER DEFAULT 0
            )
            """,
            # Index for fast time-series queries per model
            "CREATE INDEX IF NOT EXISTS idx_bm_model_batch ON batch_metrics(model_id, batch_id)",
            "CREATE INDEX IF NOT EXISTS idx_dd_model_batch ON drift_details(model_id, batch_id)",
        ]
        # SQLite AUTOINCREMENT syntax differs from PostgreSQL SERIAL
        if self._is_pg:
            for i, s in enumerate(stmts):
                stmts[i] = (s
                    .replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
                    .replace("AUTOINCREMENT", "")
                )

        with self._conn() as conn:
            cur = conn.cursor()
            for stmt in stmts:
                cur.execute(stmt)

        logger.debug("Registry schema initialised")

    # ── Model registration ────────────────────────────────────────────────

    def register_model(
        self,
        name: str,
        version: str = "1.0.0",
        description: str = "",
        pipeline_path: str = "",
        thresholds: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Register a new model (or return existing model_id if already registered).

        Args:
            name:          Unique model name, e.g. "churn_xgboost"
            version:       Semantic version string, e.g. "1.2.0"
            description:   Human-readable purpose
            pipeline_path: Path to the saved joblib pipeline
            thresholds:    Per-model alert thresholds (overrides global defaults)
            tags:          Arbitrary key-value metadata

        Returns:
            model_id (integer primary key)
        """
        # Merge with defaults
        merged_thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        merged_tags       = tags or {}

        now = datetime.now(timezone.utc).isoformat()

        # Upsert — if (name, version) exists, just return the existing id
        existing = self._fetchall(
            "SELECT model_id FROM models WHERE name = ? AND version = ?",
            (name, version),
        )
        if existing:
            model_id = existing[0]["model_id"]
            logger.info(f"Model already registered: {name} v{version} (id={model_id})")
            return model_id

        self._execute(
            """INSERT INTO models
               (name, version, description, pipeline_path,
                thresholds, tags, registered_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                name, version, description, str(pipeline_path),
                json.dumps(merged_thresholds), json.dumps(merged_tags),
                now, STATUS_INACTIVE,
            ),
        )
        rows = self._fetchall(
            "SELECT model_id FROM models WHERE name = ? AND version = ?",
            (name, version),
        )
        model_id = rows[0]["model_id"]
        logger.info(f"Registered model: {name} v{version} (id={model_id})")
        return model_id

    def update_model_status(self, model_id: int, status: str) -> None:
        """Set the status field for a model (healthy / warning / critical / inactive)."""
        self._execute(
            "UPDATE models SET status = ? WHERE model_id = ?",
            (status, model_id),
        )

    # ── Metric logging ────────────────────────────────────────────────────

    def log_batch_metrics(
        self,
        model_id: int,
        batch_id: int,
        auc_roc: float = 0.0,
        f1_score: float = 0.0,
        precision_val: float = 0.0,
        recall_val: float = 0.0,
        brier_score: float = 0.0,
        drift_score: float = 0.0,
        drift_severity: str = "none",
        n_drifted_feats: int = 0,
        n_samples: int = 0,
        retrain_triggered: bool = False,
        roi_monthly_loss: float = 0.0,
        roi_missed_churners: int = 0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Persist one batch's worth of metrics for a given model."""
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            """INSERT INTO batch_metrics
               (model_id, batch_id, recorded_at,
                auc_roc, f1_score, precision_val, recall_val, brier_score,
                drift_score, drift_severity, n_drifted_feats, n_samples,
                retrain_triggered, roi_monthly_loss, roi_missed_churners, extra_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                model_id, batch_id, now,
                round(auc_roc, 6), round(f1_score, 6),
                round(precision_val, 6), round(recall_val, 6),
                round(brier_score, 6), round(drift_score, 6),
                drift_severity, n_drifted_feats, n_samples,
                int(retrain_triggered),
                round(roi_monthly_loss, 2), roi_missed_churners,
                json.dumps(extra or {}),
            ),
        )
        # Derive model status from latest metrics
        thresholds = self._get_thresholds(model_id)
        if auc_roc < thresholds.get("auc_critical", 0.70) or drift_severity == "critical":
            status = STATUS_CRITICAL
        elif auc_roc < thresholds.get("auc_warning", 0.80) or drift_severity == "warning":
            status = STATUS_WARNING
        else:
            status = STATUS_HEALTHY
        self.update_model_status(model_id, status)

    def log_drift_details(
        self,
        model_id: int,
        batch_id: int,
        drift_results: list,   # list of DriftResult objects
    ) -> None:
        """Persist per-feature drift scores for a batch."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            cur = conn.cursor()
            ph  = self._placeholder()
            for r in drift_results:
                cur.execute(
                    f"""INSERT INTO drift_details
                        (model_id, batch_id, feature, method,
                         score, p_value, severity, is_drifted)
                        VALUES ({ph},{ph},{ph},{ph},{ph},{ph},{ph},{ph})""",
                    (
                        model_id, batch_id,
                        r.feature, r.method,
                        round(r.score, 6) if r.score is not None else 0.0,
                        round(r.p_value, 6) if r.p_value is not None else None,
                        r.severity, int(r.is_drifted),
                    ),
                )

    def log_retrain_event(
        self,
        model_id: int,
        trigger_batch: int,
        trigger_reason: str,
        old_auc: float,
        new_auc: float,
        promoted: bool,
        promotion_reason: str,
        mlflow_run_id: str = "",
    ) -> None:
        """Record a retraining event for a model."""
        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            """INSERT INTO retrain_log
               (model_id, triggered_at, trigger_batch, trigger_reason,
                old_auc, new_auc, promoted, promotion_reason, mlflow_run_id)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                model_id, now, trigger_batch, trigger_reason,
                round(old_auc, 6), round(new_auc, 6),
                int(promoted), promotion_reason, mlflow_run_id,
            ),
        )

    # ── Query methods ─────────────────────────────────────────────────────

    def get_fleet_health(self) -> pd.DataFrame:
        """
        Return one row per registered model with its latest health status.

        Columns: model_id, name, version, status, latest_auc, latest_f1,
                 drift_severity, last_batch_id, last_updated, n_batches,
                 n_retrains, roi_monthly_loss, description
        """
        rows = self._fetchall("""
            SELECT
                m.model_id,
                m.name,
                m.version,
                m.status,
                m.description,
                b.auc_roc       AS latest_auc,
                b.f1_score      AS latest_f1,
                b.drift_severity,
                b.batch_id      AS last_batch_id,
                b.recorded_at   AS last_updated,
                b.roi_monthly_loss,
                b.roi_missed_churners,
                counts.n_batches,
                COALESCE(rcount.n_retrains, 0) AS n_retrains
            FROM models m
            LEFT JOIN batch_metrics b
                ON b.id = (
                    SELECT id FROM batch_metrics
                    WHERE model_id = m.model_id
                    ORDER BY batch_id DESC LIMIT 1
                )
            LEFT JOIN (
                SELECT model_id, COUNT(*) AS n_batches
                FROM batch_metrics
                GROUP BY model_id
            ) counts ON counts.model_id = m.model_id
            LEFT JOIN (
                SELECT model_id, COUNT(*) AS n_retrains
                FROM retrain_log
                GROUP BY model_id
            ) rcount ON rcount.model_id = m.model_id
            ORDER BY m.model_id
        """)
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_model_history(
        self, model_id: int, last_n_batches: Optional[int] = None
    ) -> pd.DataFrame:
        """Full batch metric time-series for a single model."""
        limit_clause = f"LIMIT {last_n_batches}" if last_n_batches else ""
        rows = self._fetchall(
            f"""SELECT * FROM batch_metrics
                WHERE model_id = ?
                ORDER BY batch_id DESC {limit_clause}""",
            (model_id,),
        )
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        if not df.empty:
            df = df.sort_values("batch_id").reset_index(drop=True)
        return df

    def get_model_drift_history(
        self, model_id: int, batch_id: Optional[int] = None
    ) -> pd.DataFrame:
        """Per-feature drift details for a model, optionally filtered to one batch."""
        if batch_id is not None:
            rows = self._fetchall(
                "SELECT * FROM drift_details WHERE model_id = ? AND batch_id = ?",
                (model_id, batch_id),
            )
        else:
            rows = self._fetchall(
                "SELECT * FROM drift_details WHERE model_id = ? ORDER BY batch_id",
                (model_id,),
            )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_retrain_history(self, model_id: int) -> pd.DataFrame:
        """All retraining events for a model."""
        rows = self._fetchall(
            "SELECT * FROM retrain_log WHERE model_id = ? ORDER BY triggered_at",
            (model_id,),
        )
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def list_models(self) -> pd.DataFrame:
        """All registered models with their metadata."""
        rows = self._fetchall("SELECT * FROM models ORDER BY model_id")
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_thresholds(self, model_id: int) -> Dict[str, float]:
        rows = self._fetchall(
            "SELECT thresholds FROM models WHERE model_id = ?", (model_id,)
        )
        if not rows:
            return DEFAULT_THRESHOLDS
        try:
            return json.loads(rows[0]["thresholds"])
        except (json.JSONDecodeError, KeyError):
            return DEFAULT_THRESHOLDS

    def export_fleet_json(self, path: Optional[Path] = None) -> Path:
        """
        Export current fleet health to a JSON file the dashboard can read.
        Called at the end of each monitoring run.
        """
        if path is None:
            path = MODELS_DIR / "fleet_health.json"
        df = self.get_fleet_health()
        df.to_json(path, orient="records", indent=2)
        logger.info(f"Fleet health exported → {path}")
        return path
