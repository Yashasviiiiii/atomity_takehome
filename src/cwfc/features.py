"""Feature engineering for the CWFC forecasting targets."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS    = [1, 2, 3]
WINDOWS = [3, 6]
CATEGORIES = ["Delay-insensitive", "Interactive", "Unknown"]
CORE_MAP   = {"2": 2, "4": 4, "8": 8, "24": 24, ">24": 32}


def build_category_ts(cpu: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp mean avg_cpu for each VM category."""
    merged = cpu[["timestamp", "vm_id", "avg_cpu"]].merge(
        meta[["vm_id", "vm_category"]], on="vm_id", how="left"
    )
    return (
        merged.groupby(["timestamp", "vm_category"])["avg_cpu"]
        .mean()
        .unstack("vm_category")
        .rename(columns=lambda c: f"cpu_{c}")
        .reset_index()
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def build_fleet_ts(cpu: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-timestamp core-weighted CPU from cpu x meta merge."""
    enriched = cpu[["timestamp", "vm_id", "avg_cpu"]].merge(
        meta[["vm_id", "virtual_core_bucket"]], on="vm_id", how="left"
    )
    enriched["core_count"] = enriched["virtual_core_bucket"].map(CORE_MAP).fillna(4)

    records = []
    for t, g in enriched.groupby("timestamp"):
        records.append({
            "timestamp"        : t,
            "core_weighted_cpu": float(np.average(g["avg_cpu"], weights=g["core_count"])),
        })
    return pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)


def build_features(
    targets:  pd.DataFrame,
    cat_ts:   pd.DataFrame,
    meta:     pd.DataFrame,
    fleet_ts: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Lean, causal feature matrix for aggregate_cpu_utilization.
    Keeps only the high-impact short lags, rolling means, trend difference,
    and lagged sub-fleet signals.
    """
    col = "aggregate_cpu_utilization"
    df = targets[["timestamp", col]].copy()

    for lag in LAGS:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df[f"{col}_diff1"] = df[f"{col}_lag1"] - df[f"{col}_lag2"]

    for w in WINDOWS:
        df[f"{col}_roll{w}_mean"] = df[col].shift(1).rolling(w).mean()

    for cat in CATEGORIES:
        c_col = f"cpu_{cat}"
        if c_col in cat_ts.columns:
            df[f"{c_col}_lag1"] = cat_ts[c_col].shift(1)

    if fleet_ts is not None and "core_weighted_cpu" in fleet_ts.columns:
        df["core_weighted_cpu_lag1"] = fleet_ts["core_weighted_cpu"].shift(1)

    return df


def split_features(
    df: pd.DataFrame,
    train_obs: int = 110,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split; NaN warm-up rows trimmed from training only."""
    target_col   = "aggregate_cpu_utilization"
    feature_cols = [c for c in df.columns if c not in ("timestamp", target_col)]

    train_df = df.iloc[:train_obs].dropna(subset=feature_cols)
    test_df  = df.iloc[train_obs:]

    return (
        train_df[feature_cols], test_df[feature_cols],
        train_df[target_col],   test_df[target_col],
    )

