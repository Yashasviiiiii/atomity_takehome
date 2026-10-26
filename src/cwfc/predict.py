import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

from cwfc.data import load_raw, build_targets
from cwfc.features import build_category_ts, build_fleet_ts, build_features, split_features


def main():
    parser = argparse.ArgumentParser(description="Generate 24-step workload forecast")
    parser.add_argument("--data", default="challenge_data", help="Path to data folder")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    args = parser.parse_args()

    meta, cpu = load_raw(Path(args.data))
    targets = build_targets(cpu)
    cat_ts = build_category_ts(cpu, meta)
    fleet_ts = build_fleet_ts(cpu, meta)
    feat_df = build_features(targets, cat_ts, meta, fleet_ts=fleet_ts)

    train_obs = 110
    X_train, X_test, y_train, _ = split_features(feat_df, train_obs)

    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=200,
        learning_rate=0.03,
        num_leaves=8,
        min_child_samples=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        verbose=-1,
    )
    model.fit(X_train, y_train)

    preds = pd.DataFrame({
        "timestamp": targets.iloc[train_obs:]["timestamp"].values,
        "active_vm_count": 10000,
        "aggregate_cpu_utilization": np.clip(model.predict(X_test), 0, 100),
    })

    preds.to_csv(args.output, index=False)
    print(f"Generated {len(preds)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
