"""
battery_dataset_adapter.py — Real Battery Dataset Normalizer

Converts ANY real battery dataset (from your data.zip, CALCE, NASA, LG/Samsung,
Oxford, RWTH, Panasonic, etc.) into the standard schema the BMS project expects.

Standard schema
───────────────
  Voltage_V           float   terminal voltage
  Current_A           float   charge(+) / discharge(-) current
  Temperature_C       float   cell surface temperature
  SoC                 float   State of Charge  [0, 1]    ← target for LSTM
  soh                 float   State of Health  [0, 1]    ← target for SOHRegressor
  fault_label         int     0 = normal, 1 = fault/anomaly
  cycle_count         int     cumulative cycle number
  capacity_fade_pct   float   (1-soh)*100
  calendar_age_days   int     days since cell manufacturing
  internal_resistance_mohm float
  charge_rate_c       float   C-rate
  soc                 float   alias of SoC (lowercase, used by BMS models)
  cell_id             str

Common raw dataset column-name mappings
───────────────────────────────────────
CALCE:    Voltage(V)  Current(A)  Temperature (C)  SOC
NASA:     Voltage_measured  Current_measured  Temperature_measured
LG/Samsung: V  I  T  SOC_ZHU / SOC
Oxford:   Voltage  Current  Temperature
RWTH:     U  I  T  SOC

Usage
─────
  from src.data.battery_dataset_adapter import BatteryDatasetAdapter

  # Auto-detect and normalize ANY battery CSV:
  df = BatteryDatasetAdapter.load_and_normalize("data/battery_data.csv")

  # Or step by step:
  raw = pd.read_csv("data/battery_data.csv")
  df  = BatteryDatasetAdapter.normalize(raw)

  # Register into DatasetRegistry:
  BatteryDatasetAdapter.register(
      "data/battery_data.csv",
      name="my_battery_dataset",
      description="My real EV battery data",
  )
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

# ── Standard output column names ─────────────────────────────────────────────
STANDARD_COLUMNS = [
    "Voltage_V", "Current_A", "Temperature_C", "SoC",
    "soh", "fault_label", "cycle_count", "capacity_fade_pct",
    "calendar_age_days", "internal_resistance_mohm", "charge_rate_c",
    "soc", "cell_id",
]

# ── Column-name alias maps (raw name → standard name) ────────────────────────
# Each entry: list of possible raw names that map to the standard name.
COLUMN_ALIASES: Dict[str, List[str]] = {
    "Voltage_V": [
        "voltage", "voltage_v", "v", "vt", "volt",
        "voltage(v)", "voltage_measured", "terminal_voltage",
        "pack_voltage", "cell_voltage", "u", "ux",
    ],
    "Current_A": [
        "current", "current_a", "i", "current(a)", "current_measured",
        "pack_current", "cell_current", "i_a", "current_ma",  # will /1000
    ],
    "Temperature_C": [
        "temperature", "temperature_c", "temp", "t", "temp_c",
        "temperature (c)", "temperature_measured", "t_c",
        "cell_temp", "surface_temp", "ambient_temp",
    ],
    "SoC": [
        "soc", "soc_true", "soc_zhhu", "soc_zhu", "state_of_charge",
        "soc_ref", "soc_label", "reference_soc", "true_soc", "rel_soc",
        "capacity_soc", "soc(%)",   # will /100
    ],
    "soh": [
        "soh", "state_of_health", "health", "capacity_ratio",
        "relative_capacity", "normalized_capacity",
    ],
    "cycle_count": [
        "cycle", "cycle_count", "cycle_number", "cycles",
        "charge_cycle", "cycle_index", "cyc",
    ],
    "internal_resistance_mohm": [
        "resistance", "internal_resistance", "r_int", "rint",
        "internal_resistance_mohm", "ohmic_resistance",
    ],
    "charge_rate_c": [
        "c_rate", "charge_rate", "charge_rate_c", "crate",
    ],
}


class BatteryDatasetAdapter:
    """
    Normalizes any battery CSV/DataFrame into the standard BMS project schema.

    The adapter:
    1. Auto-detects column names via fuzzy alias matching
    2. Scales units (mA → A, SoC 0–100 → 0–1, etc.)
    3. Derives missing columns from available ones
       - SoC from cumulative charge (Coulomb counting) if not present
       - SOH from max capacity across cycles if not present
       - fault_label from voltage + temperature thresholds
    4. Validates the result and reports what was found / derived / missing
    """

    # ── Public API ────────────────────────────────────────────────────────

    @classmethod
    def load_and_normalize(
        cls,
        path:              str | Path,
        cell_id:           str = "cell_001",
        nominal_capacity:  float = 2.5,    # Ah — used for Coulomb counting SoC
        voltage_min:       float = 2.5,    # V  — fault threshold
        voltage_max:       float = 4.2,    # V
        temp_max:          float = 45.0,   # °C — fault threshold
    ) -> pd.DataFrame:
        """
        Load a CSV/Excel file and return a fully normalized DataFrame.

        This is the one-line entry point:
            df = BatteryDatasetAdapter.load_and_normalize("data/my_data.csv")
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")

        logger.info(f"Loading battery dataset: {path.name}")

        if path.suffix.lower() in (".xlsx", ".xls"):
            raw = pd.read_excel(path)
        elif path.suffix.lower() == ".parquet":
            raw = pd.read_parquet(path)
        elif path.suffix.lower() == ".tsv":
            raw = pd.read_csv(path, sep="\t")
        else:
            raw = pd.read_csv(path)

        logger.info(f"  Raw shape: {raw.shape}  columns: {list(raw.columns[:8])}")
        return cls.normalize(
            raw, cell_id=cell_id,
            nominal_capacity=nominal_capacity,
            voltage_min=voltage_min, voltage_max=voltage_max,
            temp_max=temp_max,
        )

    @classmethod
    def normalize(
        cls,
        df:                pd.DataFrame,
        cell_id:           str   = "cell_001",
        nominal_capacity:  float = 2.5,
        voltage_min:       float = 2.5,
        voltage_max:       float = 4.2,
        temp_max:          float = 45.0,
    ) -> pd.DataFrame:
        """
        Normalize a raw DataFrame to the standard BMS schema.
        Returns a clean DataFrame with standard column names.
        """
        df   = df.copy()
        cols = {c.lower().strip(): c for c in df.columns}
        out  = pd.DataFrame(index=df.index)

        report = {"mapped": [], "derived": [], "missing": []}

        # ── Step 1: Map known columns via alias ───────────────────────────
        for std_col, aliases in COLUMN_ALIASES.items():
            found = cls._find_column(cols, aliases)
            if found:
                out[std_col] = df[found].values.astype(np.float32)
                report["mapped"].append(f"{found} → {std_col}")
            else:
                report["missing"].append(std_col)

        # ── Step 2: Unit corrections ──────────────────────────────────────
        # mA → A
        if "Current_A" in out and out["Current_A"].abs().max() > 200:
            out["Current_A"] /= 1000.0
            logger.info("  Current_A: detected mA — converted to A")

        # SoC 0–100 → 0–1
        if "SoC" in out and out["SoC"].max() > 1.5:
            out["SoC"] /= 100.0
            logger.info("  SoC: detected percentage — converted to [0,1]")

        # SOH 0–100 → 0–1
        if "soh" in out and out["soh"].max() > 1.5:
            out["soh"] /= 100.0

        # ── Step 3: Derive SoC from Coulomb counting if missing ───────────
        if "SoC" not in out or out["SoC"].isnull().all():
            if "Current_A" in out:
                out["SoC"] = cls._coulomb_count_soc(
                    out["Current_A"].values, nominal_capacity
                )
                report["derived"].append("SoC (Coulomb counting)")
            else:
                out["SoC"] = 0.5   # neutral fallback
                report["missing"].append("SoC (no current for Coulomb counting)")

        # ── Step 4: Clip SoC to [0,1] ─────────────────────────────────────
        out["SoC"] = out["SoC"].clip(0.0, 1.0)

        # ── Step 5: Derive SOH if missing ─────────────────────────────────
        if "soh" not in out:
            if "cycle_count" in out and out["cycle_count"].max() > 0:
                # Simple linear degradation proxy
                max_cycles = out["cycle_count"].max()
                out["soh"] = (1.0 - 0.0002 * out["cycle_count"]).clip(0.5, 1.0)
                report["derived"].append("soh (linear from cycle_count)")
            else:
                out["soh"] = 1.0 - (1.0 - out["SoC"]) * 0.1
                out["soh"] = out["soh"].clip(0.5, 1.0)
                report["derived"].append("soh (proxy from SoC)")

        # ── Step 6: Derive fault_label ────────────────────────────────────
        fault = pd.Series(0, index=df.index)
        if "Voltage_V" in out:
            fault |= (out["Voltage_V"] < voltage_min).astype(int)
            fault |= (out["Voltage_V"] > voltage_max).astype(int)
        if "Temperature_C" in out:
            fault |= (out["Temperature_C"] > temp_max).astype(int)
        fault |= (out["soh"] < 0.70).astype(int)
        out["fault_label"] = fault.astype(int)
        report["derived"].append("fault_label (voltage+temp+soh thresholds)")

        # ── Step 7: Fill remaining BMS columns ────────────────────────────
        if "cycle_count" not in out:
            out["cycle_count"] = 0
        if "calendar_age_days" not in out:
            out["calendar_age_days"] = (out["cycle_count"] * 1.5).astype(int)
        if "internal_resistance_mohm" not in out:
            out["internal_resistance_mohm"] = 10.0 + out["cycle_count"] * 0.0003
        if "charge_rate_c" not in out:
            if "Current_A" in out:
                out["charge_rate_c"] = (out["Current_A"].abs() / nominal_capacity).clip(0.1, 5.0)
            else:
                out["charge_rate_c"] = 0.5

        out["capacity_fade_pct"] = ((1.0 - out["soh"]) * 100).clip(0.0, 100.0)
        out["soc"]     = out["SoC"]          # lowercase alias for BMS models
        out["cell_id"] = cell_id

        # ── Step 8: Fill nulls ─────────────────────────────────────────────
        num_cols = out.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if out[col].isnull().any():
                out[col] = out[col].fillna(out[col].median())

        # ── Report ─────────────────────────────────────────────────────────
        logger.info(f"  Mapped:  {report['mapped']}")
        logger.info(f"  Derived: {report['derived']}")
        if report["missing"]:
            logger.warning(f"  Missing (set to defaults): {report['missing']}")
        logger.success(
            f"  Normalized: {len(out)} rows × {len(out.columns)} cols  "
            f"SoC=[{out['SoC'].min():.2f},{out['SoC'].max():.2f}]  "
            f"fault_rate={out['fault_label'].mean():.3f}"
        )
        return out

    @classmethod
    def register(
        cls,
        path:        str | Path,
        name:        str,
        description: str = "",
        version:     str = "1.0.0",
        cell_id:     str = "cell_001",
        **normalize_kwargs,
    ) -> Tuple[int, pd.DataFrame]:
        """
        One-step: normalize a raw dataset and register it in DatasetRegistry.

        Returns (dataset_id, normalized_df).
        """
        from src.data.dataset_registry import DatasetRegistry

        df         = cls.load_and_normalize(path, cell_id=cell_id, **normalize_kwargs)
        registry   = DatasetRegistry()
        dataset_id = registry.register_from_dataframe(
            df,
            name=name,
            version=version,
            domain="battery",
            description=description or f"Battery dataset from {Path(path).name}",
            tags={"source": str(Path(path).name), "cell_id": cell_id},
            target_column="SoC",
        )
        logger.success(f"Registered dataset '{name}' — id={dataset_id}")
        return dataset_id, df

    @classmethod
    def split_by_cycle(
        cls,
        df:            pd.DataFrame,
        train_frac:    float = 0.70,
        val_frac:      float = 0.15,
        cycle_col:     str   = "cycle_count",
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split a battery dataset by cycle number (not random rows).
        This prevents data leakage — cycles used for training never appear in test.

        Returns (df_train, df_val, df_test).
        """
        if cycle_col in df.columns:
            cycles      = sorted(df[cycle_col].unique())
            n           = len(cycles)
            n_train     = int(n * train_frac)
            n_val       = int(n * val_frac)
            train_cyc   = set(cycles[:n_train])
            val_cyc     = set(cycles[n_train: n_train + n_val])
            test_cyc    = set(cycles[n_train + n_val:])
            df_train    = df[df[cycle_col].isin(train_cyc)]
            df_val      = df[df[cycle_col].isin(val_cyc)]
            df_test     = df[df[cycle_col].isin(test_cyc)]
        else:
            # Fall back to row-wise split preserving time order
            n           = len(df)
            n_train     = int(n * train_frac)
            n_val       = int(n * val_frac)
            df_train    = df.iloc[:n_train]
            df_val      = df.iloc[n_train: n_train + n_val]
            df_test     = df.iloc[n_train + n_val:]

        logger.info(
            f"Cycle split — train: {len(df_train):,}  "
            f"val: {len(df_val):,}  test: {len(df_test):,}"
        )
        return df_train, df_val, df_test

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _find_column(cols_lower: Dict[str, str], aliases: List[str]) -> Optional[str]:
        """Find the original column name matching any alias (case-insensitive)."""
        for alias in aliases:
            if alias.lower() in cols_lower:
                return cols_lower[alias.lower()]
        # Fuzzy: check if any column contains the alias as a substring
        for alias in aliases:
            for col_lower, col_orig in cols_lower.items():
                if alias.lower() in col_lower:
                    return col_orig
        return None

    @staticmethod
    def _coulomb_count_soc(
        current_a: np.ndarray,
        nominal_capacity: float,
        initial_soc: float = 1.0,
        dt_s: float = 1.0,
    ) -> np.ndarray:
        """
        Estimate SoC via Coulomb counting integration.
        current_a: positive = charging, negative = discharging.
        """
        capacity_as  = nominal_capacity * 3600.0   # Ah → As (coulombs)
        delta_q      = current_a * dt_s             # ΔQ per timestep
        cumulative_q = np.cumsum(delta_q)
        soc          = initial_soc + cumulative_q / capacity_as
        return np.clip(soc, 0.0, 1.0).astype(np.float32)
