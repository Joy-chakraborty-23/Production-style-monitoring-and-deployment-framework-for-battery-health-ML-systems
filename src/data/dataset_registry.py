"""
dataset_registry.py — Multi-Dataset Registry

Manages a catalogue of datasets used by models in the BMS monitoring system.
Supports:
  - Registering datasets from CSV files or DataFrames
  - Automatic schema validation and column type inference
  - Class balance reporting for classification targets
  - Dataset versioning (name + version → unique entry)
  - Querying datasets by domain, chemistry, or tag
  - Automatic augmentation (noise injection, oversampling)

Database schema:
    datasets       — catalogue entry per dataset
    dataset_stats  — per-column statistics snapshot at registration time

Usage:
    registry = DatasetRegistry()

    # Register from CSV file
    registry.register_from_file(
        path="data/bms_reference_NMC.csv",
        name="bms_nmc_reference",
        domain="battery",
        description="NMC reference dataset — 300 cells × 100 cycles",
        tags={"chemistry": "NMC", "source": "lab"},
    )

    # Register from DataFrame
    registry.register_from_dataframe(df, name="custom_dataset", domain="battery")

    # List all datasets
    registry.list_datasets()

    # Load a dataset back as DataFrame
    df = registry.load_dataset("bms_nmc_reference")
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DATA_DIR     = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"   # physical CSV copies live here
DB_PATH      = BASE_DIR / "models" / "dataset_registry.db"

for _d in [DATA_DIR, DATASETS_DIR, DB_PATH.parent]:
    _d.mkdir(parents=True, exist_ok=True)


# ── Status codes ───────────────────────────────────────────────────────────────
STATUS_ACTIVE   = "active"
STATUS_ARCHIVED = "archived"
STATUS_INVALID  = "invalid"

VALID_DOMAINS = {"battery", "churn", "ltv", "propensity", "custom"}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class DatasetEntry:
    dataset_id:   int
    name:         str
    version:      str
    domain:       str
    description:  str
    file_path:    str     # path to the stored CSV copy
    n_rows:       int
    n_cols:       int
    columns:      List[str]
    column_types: Dict[str, str]
    tags:         Dict[str, str]
    registered_at: str
    status:       str = STATUS_ACTIVE
    target_column: Optional[str] = None
    class_balance: Optional[Dict[str, float]] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["columns"]      = json.loads(d["columns"]) if isinstance(d["columns"], str) else d["columns"]
        d["column_types"] = json.loads(d["column_types"]) if isinstance(d["column_types"], str) else d["column_types"]
        return d


# ── Registry class ─────────────────────────────────────────────────────────────

class DatasetRegistry:
    """
    Central catalogue of all datasets available to the monitoring system.

    Every dataset is:
      1. Copied to data/datasets/<name>_v<version>.csv  (immutable snapshot)
      2. Indexed in a SQLite catalogue with schema + statistics
      3. Queryable by name, domain, tags, or column schema

    This ensures reproducibility — models always load the exact data
    version they were trained on, even if the source file changes.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"DatasetRegistry initialised — {self._db_path}")

    # ── DB helpers ─────────────────────────────────────────────────────────

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _execute(self, sql: str, params: tuple = ()) -> None:
        with self._conn() as conn:
            conn.cursor().execute(sql, params)

    def _fetchall(self, sql: str, params: tuple = ()) -> List[dict]:
        with self._conn() as conn:
            cur = conn.cursor()
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    def _init_schema(self) -> None:
        stmts = [
            """
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                name          TEXT NOT NULL,
                version       TEXT NOT NULL DEFAULT '1.0.0',
                domain        TEXT NOT NULL DEFAULT 'custom',
                description   TEXT DEFAULT '',
                file_path     TEXT NOT NULL,
                n_rows        INTEGER DEFAULT 0,
                n_cols        INTEGER DEFAULT 0,
                columns       TEXT DEFAULT '[]',
                column_types  TEXT DEFAULT '{}',
                tags          TEXT DEFAULT '{}',
                registered_at TEXT NOT NULL,
                status        TEXT DEFAULT 'active',
                target_column TEXT,
                class_balance TEXT,
                UNIQUE(name, version)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS dataset_stats (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id  INTEGER NOT NULL REFERENCES datasets(dataset_id),
                column_name TEXT NOT NULL,
                dtype       TEXT,
                mean        REAL,
                std         REAL,
                min_val     REAL,
                max_val     REAL,
                p25         REAL,
                p50         REAL,
                p75         REAL,
                null_count  INTEGER DEFAULT 0,
                unique_count INTEGER DEFAULT 0,
                top_values  TEXT DEFAULT '[]'
            )
            """,
            "CREATE INDEX IF NOT EXISTS idx_ds_name ON datasets(name)",
            "CREATE INDEX IF NOT EXISTS idx_ds_domain ON datasets(domain)",
        ]
        with self._conn() as conn:
            cur = conn.cursor()
            for s in stmts:
                cur.execute(s)
        logger.debug("DatasetRegistry schema initialised")

    # ── Registration ───────────────────────────────────────────────────────

    def register_from_file(
        self,
        path:          str | Path,
        name:          str,
        version:       str = "1.0.0",
        domain:        str = "custom",
        description:   str = "",
        tags:          Optional[Dict[str, str]] = None,
        target_column: Optional[str] = None,
        overwrite:     bool = False,
    ) -> int:
        """
        Register a CSV / Excel file as a dataset.

        The file is copied to data/datasets/ for immutability.
        Returns dataset_id.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")

        # Load
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

        return self.register_from_dataframe(
            df, name=name, version=version, domain=domain,
            description=description, tags=tags,
            target_column=target_column,
            source_filename=path.name,
            overwrite=overwrite,
        )

    def register_from_dataframe(
        self,
        df:             pd.DataFrame,
        name:           str,
        version:        str = "1.0.0",
        domain:         str = "custom",
        description:    str = "",
        tags:           Optional[Dict[str, str]] = None,
        target_column:  Optional[str] = None,
        source_filename: str = "",
        overwrite:      bool = False,
    ) -> int:
        """
        Register a pandas DataFrame directly.

        Steps:
          1. Validate schema (no empty column names, sensible dtypes)
          2. Save an immutable CSV snapshot to data/datasets/
          3. Compute and store per-column statistics
          4. Write catalogue entry to DB
        """
        # ── Check for existing registration ─────────────────────────────
        existing = self._fetchall(
            "SELECT dataset_id FROM datasets WHERE name=? AND version=?",
            (name, version)
        )
        if existing and not overwrite:
            did = existing[0]["dataset_id"]
            logger.info(f"Dataset already registered: {name} v{version} (id={did})")
            return did
        elif existing and overwrite:
            self._execute(
                "DELETE FROM datasets WHERE name=? AND version=?", (name, version)
            )

        # ── Validate ─────────────────────────────────────────────────────
        df = self._validate_and_clean(df, name)

        # ── Save CSV snapshot ─────────────────────────────────────────────
        safe_name  = name.replace(" ", "_").replace("/", "_")
        csv_name   = f"{safe_name}_v{version}.csv"
        dest_path  = DATASETS_DIR / csv_name
        df.to_csv(dest_path, index=False)

        # ── Class balance ─────────────────────────────────────────────────
        class_balance = None
        if target_column and target_column in df.columns:
            vc = df[target_column].value_counts(normalize=True).round(4)
            class_balance = vc.to_dict()

        # ── Column types ──────────────────────────────────────────────────
        col_types = {c: str(df[c].dtype) for c in df.columns}

        now = datetime.now(timezone.utc).isoformat()
        self._execute(
            """INSERT INTO datasets
               (name, version, domain, description, file_path,
                n_rows, n_cols, columns, column_types, tags,
                registered_at, status, target_column, class_balance)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                name, version, domain, description, str(dest_path),
                len(df), len(df.columns),
                json.dumps(list(df.columns)),
                json.dumps(col_types),
                json.dumps(tags or {}),
                now, STATUS_ACTIVE,
                target_column,
                json.dumps(class_balance) if class_balance else None,
            )
        )

        rows = self._fetchall(
            "SELECT dataset_id FROM datasets WHERE name=? AND version=?",
            (name, version)
        )
        dataset_id = rows[0]["dataset_id"]

        # ── Per-column statistics ─────────────────────────────────────────
        self._store_column_stats(dataset_id, df)

        logger.success(
            f"Registered dataset '{name}' v{version} "
            f"(id={dataset_id}  rows={len(df)}  cols={len(df.columns)})"
        )
        return dataset_id

    # ── Loading ────────────────────────────────────────────────────────────

    def load_dataset(
        self,
        name:    str,
        version: str = "1.0.0",
    ) -> pd.DataFrame:
        """Load a registered dataset as a DataFrame."""
        rows = self._fetchall(
            "SELECT * FROM datasets WHERE name=? AND version=?",
            (name, version)
        )
        if not rows:
            raise KeyError(f"Dataset not found: {name} v{version}")
        row = rows[0]
        path = Path(row["file_path"])
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset file missing from disk: {path}\n"
                f"Re-register the dataset or restore the file."
            )
        return pd.read_csv(path)

    def load_latest(self, name: str) -> pd.DataFrame:
        """Load the most recently registered version of a dataset by name."""
        rows = self._fetchall(
            "SELECT * FROM datasets WHERE name=? ORDER BY registered_at DESC LIMIT 1",
            (name,)
        )
        if not rows:
            raise KeyError(f"No dataset named '{name}' found.")
        return pd.read_csv(Path(rows[0]["file_path"]))

    # ── Querying ───────────────────────────────────────────────────────────

    def list_datasets(
        self,
        domain: Optional[str] = None,
        status: Optional[str] = STATUS_ACTIVE,
    ) -> pd.DataFrame:
        """Return a DataFrame catalogue of all registered datasets."""
        sql    = "SELECT * FROM datasets"
        params: list = []
        clauses = []
        if domain:
            clauses.append("domain = ?")
            params.append(domain)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY registered_at DESC"
        rows = self._fetchall(sql, tuple(params))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["columns"] = df["columns"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        return df

    def get_dataset_info(self, name: str, version: str = "1.0.0") -> dict:
        """Return full metadata + per-column stats for a dataset."""
        rows = self._fetchall(
            "SELECT * FROM datasets WHERE name=? AND version=?", (name, version)
        )
        if not rows:
            raise KeyError(f"Dataset not found: {name} v{version}")
        info = dict(rows[0])
        info["columns"]      = json.loads(info["columns"])
        info["column_types"] = json.loads(info["column_types"])
        info["tags"]         = json.loads(info["tags"])
        if info.get("class_balance"):
            info["class_balance"] = json.loads(info["class_balance"])

        # Attach column statistics
        did = info["dataset_id"]
        stat_rows = self._fetchall(
            "SELECT * FROM dataset_stats WHERE dataset_id=?", (did,)
        )
        info["column_stats"] = stat_rows
        return info

    def get_column_stats(self, name: str, version: str = "1.0.0") -> pd.DataFrame:
        """Return per-column statistics for a dataset."""
        rows = self._fetchall(
            "SELECT dataset_id FROM datasets WHERE name=? AND version=?",
            (name, version)
        )
        if not rows:
            raise KeyError(f"Dataset not found: {name} v{version}")
        stat_rows = self._fetchall(
            "SELECT * FROM dataset_stats WHERE dataset_id=?",
            (rows[0]["dataset_id"],)
        )
        return pd.DataFrame(stat_rows) if stat_rows else pd.DataFrame()

    def find_compatible_datasets(
        self,
        required_columns: List[str],
        domain: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Find datasets that contain ALL of the required columns.
        Useful for automatically selecting training data for a given model.
        """
        all_ds = self.list_datasets(domain=domain)
        if all_ds.empty:
            return pd.DataFrame()
        required_set = set(required_columns)
        compatible   = []
        for _, row in all_ds.iterrows():
            ds_cols = set(row["columns"]) if isinstance(row["columns"], list) else set()
            if required_set.issubset(ds_cols):
                compatible.append(row)
        return pd.DataFrame(compatible) if compatible else pd.DataFrame()

    # ── Management ─────────────────────────────────────────────────────────

    def archive_dataset(self, name: str, version: str = "1.0.0") -> None:
        """Mark a dataset as archived (not active, but not deleted)."""
        self._execute(
            "UPDATE datasets SET status=? WHERE name=? AND version=?",
            (STATUS_ARCHIVED, name, version)
        )
        logger.info(f"Archived dataset: {name} v{version}")

    def delete_dataset(
        self,
        name:            str,
        version:         str = "1.0.0",
        delete_file:     bool = False,
    ) -> None:
        """Remove a dataset from the registry (optionally delete the CSV file)."""
        rows = self._fetchall(
            "SELECT * FROM datasets WHERE name=? AND version=?", (name, version)
        )
        if not rows:
            raise KeyError(f"Dataset not found: {name} v{version}")
        did  = rows[0]["dataset_id"]
        path = Path(rows[0]["file_path"])

        self._execute("DELETE FROM dataset_stats WHERE dataset_id=?", (did,))
        self._execute(
            "DELETE FROM datasets WHERE name=? AND version=?", (name, version)
        )
        if delete_file and path.exists():
            path.unlink()
            logger.info(f"Deleted file: {path}")
        logger.info(f"Removed dataset from registry: {name} v{version}")

    def augment_dataset(
        self,
        name:         str,
        version:      str = "1.0.0",
        noise_level:  float = 0.02,
        n_augmented:  int = 500,
        new_version:  str = "1.0.0-aug",
    ) -> int:
        """
        Create an augmented copy of a dataset by adding Gaussian noise to
        numerical columns. Useful when the dataset is small.
        Returns the new dataset_id.
        """
        df = self.load_dataset(name, version)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        rng = np.random.default_rng(42)
        aug_rows = []
        for _ in range(n_augmented):
            row = df.sample(1).copy()
            for col in num_cols:
                std = df[col].std() * noise_level
                row[col] += rng.normal(0, std)
            aug_rows.append(row)

        aug_df = pd.concat([df] + aug_rows, ignore_index=True)
        info   = self.get_dataset_info(name, version)
        return self.register_from_dataframe(
            aug_df,
            name=name,
            version=new_version,
            domain=info["domain"],
            description=f"{info['description']} [augmented +{n_augmented} rows]",
            tags=json.loads(info["tags"]) if isinstance(info["tags"], str) else info["tags"],
            target_column=info.get("target_column"),
        )

    # ── Private helpers ────────────────────────────────────────────────────

    def _validate_and_clean(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """Basic validation: drop unnamed columns, warn on high null rates."""
        # Drop unnamed columns
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        # Warn on high nulls
        null_pct = df.isnull().mean()
        high_null = null_pct[null_pct > 0.3]
        if not high_null.empty:
            logger.warning(
                f"Dataset '{name}' has columns with >30% nulls: "
                f"{high_null.to_dict()}"
            )
        # Fill numeric nulls with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        return df

    def _store_column_stats(self, dataset_id: int, df: pd.DataFrame) -> None:
        """Compute and persist per-column statistics."""
        with self._conn() as conn:
            cur = conn.cursor()
            for col in df.columns:
                series = df[col]
                dtype  = str(series.dtype)
                null_count   = int(series.isnull().sum())
                unique_count = int(series.nunique())

                if pd.api.types.is_numeric_dtype(series):
                    s = series.dropna()
                    cur.execute(
                        """INSERT INTO dataset_stats
                           (dataset_id, column_name, dtype, mean, std,
                            min_val, max_val, p25, p50, p75,
                            null_count, unique_count, top_values)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            dataset_id, col, dtype,
                            float(s.mean()) if len(s) else None,
                            float(s.std())  if len(s) else None,
                            float(s.min())  if len(s) else None,
                            float(s.max())  if len(s) else None,
                            float(s.quantile(0.25)) if len(s) else None,
                            float(s.quantile(0.50)) if len(s) else None,
                            float(s.quantile(0.75)) if len(s) else None,
                            null_count, unique_count, "[]",
                        )
                    )
                else:
                    top = series.value_counts().head(5).index.tolist()
                    cur.execute(
                        """INSERT INTO dataset_stats
                           (dataset_id, column_name, dtype,
                            null_count, unique_count, top_values)
                           VALUES (?,?,?,?,?,?)""",
                        (
                            dataset_id, col, dtype,
                            null_count, unique_count, json.dumps([str(v) for v in top])
                        )
                    )


# ── Module-level convenience instance ─────────────────────────────────────────

_default_registry: Optional[DatasetRegistry] = None


def get_dataset_registry() -> DatasetRegistry:
    """Return (or create) the default singleton DatasetRegistry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = DatasetRegistry()
    return _default_registry
