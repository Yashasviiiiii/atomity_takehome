"""Data loading and target construction for the CWFC challenge."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "challenge_data"
TRAIN_OBS = 110
TEST_OBS = 24


def load_raw(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load vm_metadata and vm_cpu_readings from data_dir."""
    meta = pd.read_csv(data_dir / "vm_metadata.csv")
    cpu = pd.read_csv(
        data_dir / "vm_cpu_readings.csv",
        dtype={"timestamp": "int64", "vm_id": "str",
               "min_cpu": "float64", "max_cpu": "float64", "avg_cpu": "float64"},
    )
    return meta, cpu


def build_targets(cpu: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate VM-level readings into the two required time-series targets.

    Returns a DataFrame indexed by timestamp (sorted ascending) with columns:
        timestamp, active_vm_count, aggregate_cpu_utilization
    """
    agg = (
        cpu.groupby("timestamp")
        .agg(
            active_vm_count=("vm_id", "nunique"),
            aggregate_cpu_utilization=("avg_cpu", "mean"),
        )
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    return agg


def train_test_split(targets: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train, test) using the 110/24 chronological split."""
    assert len(targets) == TRAIN_OBS + TEST_OBS, (
        f"Expected {TRAIN_OBS + TEST_OBS} observations, got {len(targets)}"
    )
    train = targets.iloc[:TRAIN_OBS].copy()
    test = targets.iloc[TRAIN_OBS:].copy()
    return train, test
