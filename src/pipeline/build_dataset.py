"""
End-to-end dataset assembly: load raw -> engineer features -> split by unit
(train vs internal validation) -> verify zero unit overlap -> clip RUL ->
attach binary maintenance-risk label.

The official test_FD001.txt + RUL_FD001.txt is kept as a fully separate,
untouched held-out set (never used for the train/val split below).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.pipeline import config
from src.pipeline.data_loader import load_train, load_test_with_rul
from src.pipeline.features import add_features, get_feature_columns, drop_warmup_cycles


def make_classification_label(df, horizon=config.MAINTENANCE_HORIZON):
    return (df["RUL"] <= horizon).astype(int)


def verify_no_unit_overlap(train_df, val_df, label="train/val"):
    train_units = set(train_df["unit"].unique())
    val_units = set(val_df["unit"].unique())
    overlap = train_units & val_units
    print(f"[{label} split check] train units: {len(train_units)}, "
          f"val units: {len(val_units)}, overlap: {len(overlap)}")
    assert len(overlap) == 0, f"DATA LEAKAGE: unit(s) {overlap} appear in both {label} splits!"
    print(f"[{label} split check] PASSED — zero unit_id overlap confirmed.")
    return True


def build():
    print("Loading raw data...")
    train_raw = load_train()
    test_raw = load_test_with_rul()

    print("Engineering features (rolling mean/std + slope, per-unit)...")
    train_feat = add_features(train_raw)
    test_feat = add_features(test_raw)

    # Drop warm-up cycles whose feature windows are not yet full width, so
    # the training distribution matches what the API is allowed to serve.
    n_before_tr, n_before_te = len(train_feat), len(test_feat)
    train_feat = drop_warmup_cycles(train_feat)
    test_feat = drop_warmup_cycles(test_feat)
    print(f"Dropped warm-up cycles (< cycle {config.MIN_HISTORY_CYCLES}) for "
          f"train/serve feature consistency: "
          f"train {n_before_tr}->{len(train_feat)} rows, "
          f"test {n_before_te}->{len(test_feat)} rows")

    feature_cols = get_feature_columns(train_feat)

    # --- split by unit, not by row ---
    all_units = train_feat["unit"].unique()
    train_units, val_units = train_test_split(
        all_units, test_size=config.VAL_FRACTION, random_state=config.RANDOM_STATE
    )
    tr_df = train_feat[train_feat["unit"].isin(train_units)].copy()
    val_df = train_feat[train_feat["unit"].isin(val_units)].copy()

    verify_no_unit_overlap(tr_df, val_df, label="internal train/val")

    # Note on test set: train_FD001.txt and test_FD001.txt are SEPARATE
    # files, each with its own independent run-to-failure / truncated
    # trajectories, and C-MAPSS numbers units 1..100 within each file
    # independently. A "unit 7" in train and "unit 7" in test are DIFFERENT
    # physical engines, not the same engine split across files — so a raw
    # unit-number-overlap check across files would be a false positive, not
    # a leakage signal. The only place row-level leakage could actually
    # occur is the train/val split *within* train_FD001.txt, which is what
    # verify_no_unit_overlap checks and confirms clean above. The held-out
    # test set is never mixed into train/val at any point in this pipeline.
    print(f"[train+val vs test] test set is a fully separate file "
          f"({test_feat['unit'].nunique()} independent engines) — "
          f"never combined with train/val at any pipeline step.")

    # --- RUL clipping ---
    for d in (tr_df, val_df, test_feat):
        d["RUL_clipped"] = d["RUL"].clip(upper=config.RUL_CLIP)

    # --- classification label ---
    for d in (tr_df, val_df, test_feat):
        d["needs_maintenance"] = make_classification_label(d)

    return {
        "train": tr_df,
        "val": val_df,
        "test": test_feat,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    data = build()
    for split_name in ["train", "val", "test"]:
        d = data["train"] if split_name == "train" else data[split_name]
        print(f"{split_name}: {d.shape[0]} rows, {d['unit'].nunique()} units, "
              f"maintenance-flag rate = {d['needs_maintenance'].mean():.3%}")
    print(f"\n# feature columns: {len(data['feature_cols'])}")
