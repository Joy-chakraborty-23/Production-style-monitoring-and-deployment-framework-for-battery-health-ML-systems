"""
sp20_data_preprocessor.py — SP20-2 Battery Data Preprocessor

Converts the raw Arbin SP20-2 XLS battery test files into clean CSVs
that match the feature schema expected by the SoC LSTM notebook and the
BMS monitoring system's DatasetRegistry.

Input (raw XLS from Arbin cycler):
    SP2_<temp>_<test>/<date>_SP20-2_<test>_<initSOC>.xls
    Columns: Data_Point, Test_Time(s), Current(A), Voltage(V),
             Charge_Capacity(Ah), Discharge_Capacity(Ah), dV/dt(V/s), ...

Output (processed CSV per file + one merged CSV):
    Columns:
      Voltage [V]       — from Voltage(V)
      Current [A]       — from Current(A)
      Temperature [degC]— from folder name (0 / 25 / 45)
      Power [W]         — Voltage * Current
      CC_Capacity [Ah]  — Coulomb-counted capacity (integral of current)
      SOC [-]           — State of Charge via Coulomb counting + initial SOC
      Test_Time(s)      — kept for visualisation / time-series ordering
      temperature_c     — numeric temperature label
      test_type         — DST or FUDS
      initial_soc       — 0.50 or 0.80
      source_file       — original filename (for traceability)

SOC computation:
    SOC(t) = clip(initial_soc + ∫I(t)dt / Q_nominal, 0, 1)
    where Q_nominal = 2.15 Ah (SP20-2 rated capacity)

Usage:
    # From CLI — process all files and register into DatasetRegistry:
    python -m src.data.sp20_data_preprocessor \\
        --data-dir data/raw/SP2 \\
        --output-dir data/processed \\
        --register

    # From Python:
    from src.data.sp20_data_preprocessor import SP20Preprocessor
    prep  = SP20Preprocessor(data_dir="data/raw/SP2")
    df    = prep.process_all()           # merged DataFrame
    train, test = prep.split_by_temperature(df)
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    from loguru import logger
except ImportError:
    import logging as _logging
    logger = _logging.getLogger(__name__)
    logger.success = logger.info

# ── Constants ─────────────────────────────────────────────────────────────────
Q_NOMINAL_AH  = 2.15        # SP20-2 rated capacity (Ah)
VOLTAGE_MIN   = 2.50        # Below this = fully discharged
VOLTAGE_MAX   = 4.20        # Above this = fully charged

# LSTM notebook feature names (must match exactly)
FEATURE_COLS = [
    "Voltage [V]",
    "Current [A]",
    "Temperature [degC]",
    "Power [W]",
    "CC_Capacity [Ah]",
]
LABEL_COL    = "SOC [-]"

# Folder name parsing: SP2_25C_DST → temp=25, test=DST
FOLDER_RE = re.compile(r"SP2_(\d+)C_(\w+)", re.IGNORECASE)


class SP20Preprocessor:
    """
    Loads all Arbin SP20-2 XLS files from a directory tree and converts
    them into clean feature-labelled DataFrames ready for LSTM training.

    Directory structure expected:
        data_dir/
          SP2_0C_DST/
            *_0C_DST_50SOC.xls
            *_0C_DST_80SOC.xls
          SP2_0C_FUDS/  ...
          SP2_25C_DST/  ...
          SP2_25C_FUDS/ ...
          SP2_45C_DST/  ...
          SP2_45C_FUDS/ ...
    """

    def __init__(
        self,
        data_dir:    str | Path,
        q_nominal:   float = Q_NOMINAL_AH,
        output_dir:  Optional[str | Path] = None,
    ):
        self.data_dir   = Path(data_dir)
        self.q_nominal  = q_nominal
        self.output_dir = Path(output_dir) if output_dir else self.data_dir.parent / "processed"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    # ── Public API ────────────────────────────────────────────────────────

    def process_all(self, save_individual: bool = True) -> pd.DataFrame:
        """
        Process every XLS file found under data_dir.

        Returns:
            Single merged DataFrame with all files, labelled by temperature,
            test type, initial SOC, and source filename.
        """
        xls_files = list(self.data_dir.rglob("*.xls"))
        if not xls_files:
            raise FileNotFoundError(
                f"No .xls files found under {self.data_dir}\n"
                f"Check the path and ensure the data.zip was extracted here."
            )

        logger.info(f"Found {len(xls_files)} XLS files — processing...")
        all_dfs: List[pd.DataFrame] = []

        for path in sorted(xls_files):
            try:
                df = self._process_single_file(path)
                if df is not None and len(df) > 0:
                    if save_individual:
                        out_name = path.stem + "_processed.csv"
                        out_path = self.output_dir / out_name
                        df.to_csv(out_path, index=False)
                        logger.info(f"  Saved {out_name}  ({len(df):,} rows)")
                    all_dfs.append(df)
            except Exception as e:
                logger.warning(f"  Skipped {path.name}: {e}")

        if not all_dfs:
            raise RuntimeError("No files were successfully processed.")

        merged = pd.concat(all_dfs, ignore_index=True)

        # Save merged dataset
        merged_path = self.output_dir / "sp20_all_merged.csv"
        merged.to_csv(merged_path, index=False)
        logger.success(
            f"Merged dataset: {len(merged):,} rows × {len(merged.columns)} cols  "
            f"→ {merged_path}"
        )
        self._print_summary(merged)
        return merged

    def process_single_file(self, path: str | Path) -> pd.DataFrame:
        """Public wrapper for processing one file."""
        return self._process_single_file(Path(path))

    def split_by_temperature(
        self,
        df: pd.DataFrame,
        test_temps:  List[int] = None,
        train_temps: List[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split dataset into train/test by temperature — same strategy as the notebook.

        Default: train on 25°C + 45°C, test on 0°C (hardest generalisation test).

        Returns: (train_df, test_df)
        """
        test_temps  = test_temps  or [0]
        train_temps = train_temps or [25, 45]

        train = df[df["temperature_c"].isin(train_temps)].copy()
        test  = df[df["temperature_c"].isin(test_temps)].copy()

        logger.info(
            f"Train split: {len(train):,} rows (temps={train_temps})  "
            f"Test split: {len(test):,} rows (temps={test_temps})"
        )
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def split_by_test_type(
        self,
        df: pd.DataFrame,
        test_type: str = "FUDS",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by test protocol: train on DST, test on FUDS (or vice versa).
        Returns: (train_df, test_df)
        """
        test  = df[df["test_type"] == test_type].copy()
        train = df[df["test_type"] != test_type].copy()
        logger.info(f"Train: {len(train):,}  Test ({test_type}): {len(test):,}")
        return train.reset_index(drop=True), test.reset_index(drop=True)

    def get_feature_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-column statistics for the processed dataset."""
        return df[FEATURE_COLS + [LABEL_COL]].describe().round(4)

    # ── Private: single file processing ──────────────────────────────────

    def _process_single_file(self, path: Path) -> Optional[pd.DataFrame]:
        """
        Process one XLS file:
          1. Parse temperature + test type from folder name
          2. Parse initial SOC from filename
          3. Load the data sheet
          4. Compute SOC via Coulomb counting
          5. Compute Power and CC_Capacity
          6. Rename columns to LSTM notebook schema
        """
        # ── Parse metadata from path ──────────────────────────────────────
        folder_name = path.parent.name
        match = FOLDER_RE.match(folder_name)
        if not match:
            logger.warning(f"Cannot parse temperature/test from folder: {folder_name}")
            return None

        temperature_c = int(match.group(1))   # 0, 25, or 45
        test_type     = match.group(2).upper() # DST or FUDS

        # Parse initial SOC from filename: "...50SOC.xls" → 0.50
        soc_match = re.search(r"(\d+)SOC", path.name, re.IGNORECASE)
        if not soc_match:
            logger.warning(f"Cannot parse initial SOC from filename: {path.name}")
            return None
        initial_soc = int(soc_match.group(1)) / 100.0   # 50 → 0.50, 80 → 0.80

        # ── Load data sheet ───────────────────────────────────────────────
        xl    = pd.ExcelFile(str(path))
        sheet = next((s for s in xl.sheet_names if s != "Info"), None)
        if sheet is None:
            logger.warning(f"No data sheet found in {path.name}")
            return None

        raw = xl.parse(sheet)

        # ── Validate required columns ─────────────────────────────────────
        required = ["Test_Time(s)", "Current(A)", "Voltage(V)",
                    "Charge_Capacity(Ah)", "Discharge_Capacity(Ah)"]
        missing = [c for c in required if c not in raw.columns]
        if missing:
            logger.warning(f"Missing columns in {path.name}: {missing}")
            return None

        df = raw.copy()

        # ── SOC via Coulomb counting ──────────────────────────────────────
        dt         = df["Test_Time(s)"].diff().fillna(0).clip(lower=0)
        delta_q_ah = df["Current(A)"] * dt / 3600.0      # Ah per timestep
        net_q      = delta_q_ah.cumsum()
        df["SOC [-]"] = np.clip(initial_soc + net_q / self.q_nominal, 0.0, 1.0)

        # ── Derived features ──────────────────────────────────────────────
        df["Power [W]"]        = df["Voltage(V)"] * df["Current(A)"]
        df["CC_Capacity [Ah]"] = net_q                                # signed integral of current
        df["Temperature [degC]"] = float(temperature_c)

        # ── Rename to LSTM notebook schema ────────────────────────────────
        df = df.rename(columns={
            "Voltage(V)":  "Voltage [V]",
            "Current(A)":  "Current [A]",
        })

        # ── Metadata columns ──────────────────────────────────────────────
        df["temperature_c"] = temperature_c
        df["test_type"]     = test_type
        df["initial_soc"]   = initial_soc
        df["source_file"]   = path.name

        # ── Select and order final columns ────────────────────────────────
        keep = (
            FEATURE_COLS
            + [LABEL_COL, "Test_Time(s)"]
            + ["temperature_c", "test_type", "initial_soc", "source_file"]
        )
        # Only keep columns that actually exist
        keep = [c for c in keep if c in df.columns]
        df   = df[keep].copy()

        # ── Drop rows with NaN in features or label ───────────────────────
        df = df.dropna(subset=FEATURE_COLS + [LABEL_COL]).reset_index(drop=True)

        logger.info(
            f"  {path.name:<50}  T={temperature_c:>2}°C  "
            f"{test_type:<4}  SOC0={initial_soc:.0%}  "
            f"rows={len(df):,}  SOC=[{df[LABEL_COL].min():.3f},{df[LABEL_COL].max():.3f}]"
        )
        return df

    def _print_summary(self, df: pd.DataFrame) -> None:
        """Print a dataset summary table."""
        print("\n" + "═" * 65)
        print("  SP20-2 Dataset Summary")
        print("═" * 65)
        print(f"  Total rows:   {len(df):,}")
        print(f"  Total files:  {df['source_file'].nunique()}")
        print(f"  SOC range:    {df[LABEL_COL].min():.3f} – {df[LABEL_COL].max():.3f}")
        print(f"  Voltage:      {df['Voltage [V]'].min():.3f} – {df['Voltage [V]'].max():.3f} V")
        print(f"  Current:      {df['Current [A]'].min():.3f} – {df['Current [A]'].max():.3f} A")
        print()
        print("  By Temperature:")
        for t, g in df.groupby("temperature_c"):
            print(f"    {t:>2}°C  rows={len(g):>7,}  "
                  f"SOC=[{g[LABEL_COL].min():.3f},{g[LABEL_COL].max():.3f}]")
        print()
        print("  By Test Type:")
        for t, g in df.groupby("test_type"):
            print(f"    {t:<5}  rows={len(g):>7,}  files={g['source_file'].nunique()}")
        print("═" * 65 + "\n")


# ── CLI entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SP20-2 battery XLS files for BMS monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files, save CSVs, register in DatasetRegistry:
  python -m src.data.sp20_data_preprocessor \\
      --data-dir data/raw/SP2 --output-dir data/processed --register

  # Process only, no registration:
  python -m src.data.sp20_data_preprocessor \\
      --data-dir data/raw/SP2 --output-dir data/processed
        """
    )
    parser.add_argument("--data-dir",   required=True, help="Root dir containing SP2_*C_* folders")
    parser.add_argument("--output-dir", default="data/processed", help="Where to save processed CSVs")
    parser.add_argument("--register",   action="store_true", help="Register merged dataset in DatasetRegistry")
    parser.add_argument("--q-nominal",  type=float, default=2.15, help="Rated cell capacity in Ah (default: 2.15)")
    args = parser.parse_args()

    prep   = SP20Preprocessor(args.data_dir, q_nominal=args.q_nominal, output_dir=args.output_dir)
    merged = prep.process_all(save_individual=True)

    if args.register:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from src.data.dataset_registry import DatasetRegistry
        registry   = DatasetRegistry()
        dataset_id = registry.register_from_dataframe(
            df            = merged,
            name          = "sp20_battery_soc",
            version       = "1.0.0",
            domain        = "battery",
            description   = (
                "SP20-2 Li-ion battery SoC dataset — 12 Arbin test files, "
                "3 temperatures (0/25/45°C), 2 protocols (DST/FUDS), "
                "2 initial SOCs (50%/80%). SOC computed via Coulomb counting."
            ),
            tags          = {
                "chemistry":  "Li-ion",
                "cell":       "SP20-2",
                "source":     "Arbin_cycler",
                "q_nominal":  str(args.q_nominal),
                "features":   "Voltage,Current,Temperature,Power,CC_Capacity",
            },
            target_column = "SOC [-]",
        )
        print(f"\n✅ Dataset registered as 'sp20_battery_soc' v1.0.0  (id={dataset_id})")
        print(f"   Load with: DatasetRegistry().load_dataset('sp20_battery_soc')")


if __name__ == "__main__":
    main()
