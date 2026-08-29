"""
Loads raw C-MAPSS FD001 files and computes RUL labels for training data.
"""
import pandas as pd
from src.pipeline import config


def _load_raw(path):
    df = pd.read_csv(path, sep=r"\s+", header=None, names=config.ALL_COLS)
    return df


def load_train():
    """Load train_FD001.txt and attach RUL = (max cycle for unit) - cycle."""
    df = _load_raw(config.TRAIN_FILE)
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = max_cycle - df["cycle"]
    return df


def load_test_with_rul():
    """Load test_FD001.txt and attach true RUL from RUL_FD001.txt.

    RUL_FD001.txt gives the RUL at the LAST cycle recorded for each test
    unit (test trajectories are truncated before failure). For any earlier
    cycle within a unit, true RUL = (rows remaining for that unit) + final_rul.
    """
    df = _load_raw(config.TEST_FILE)
    final_rul = pd.read_csv(config.RUL_FILE, sep=r"\s+", header=None, names=["final_RUL"])
    final_rul["unit"] = final_rul.index + 1  # 1-indexed unit numbers

    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df = df.merge(final_rul, on="unit", how="left")
    df["RUL"] = (max_cycle - df["cycle"]) + df["final_RUL"]
    df = df.drop(columns=["final_RUL"])
    return df


if __name__ == "__main__":
    train = load_train()
    test = load_test_with_rul()
    print("Train shape:", train.shape, "| units:", train['unit'].nunique())
    print("Test shape:", test.shape, "| units:", test['unit'].nunique())
    print(train[["unit", "cycle", "RUL"]].head())
    print(test[["unit", "cycle", "RUL"]].tail())
