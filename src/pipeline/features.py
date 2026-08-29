"""
Feature engineering: rolling statistics + degradation-trend slope per sensor,
computed strictly within each engine unit's own cycle history (no leakage
across units, no use of future cycles at any given row).
"""
import numpy as np
import pandas as pd
from src.pipeline import config


def _kept_sensor_cols():
    return [c for c in config.SENSOR_COLS if c not in config.DROP_SENSORS]


def _rolling_slope(series, window):
    """Least-squares slope of `series` over a trailing window (cycles as x)."""
    x = np.arange(window)
    x_mean = x.mean()
    denom = ((x - x_mean) ** 2).sum()

    def slope_fn(vals):
        y = vals.values
        y_mean = y.mean()
        return ((x - x_mean) * (y - y_mean)).sum() / denom

    return series.rolling(window, min_periods=window).apply(slope_fn, raw=False)


def add_features(df):
    """Add rolling mean/std (per configured window) and a degradation slope
    feature for each kept sensor, grouped by unit so no engine's window
    ever reaches into another engine's rows.
    """
    df = df.sort_values(["unit", "cycle"]).reset_index(drop=True)
    sensors = _kept_sensor_cols()

    grouped = df.groupby("unit", sort=False)

    for w in config.ROLLING_WINDOWS:
        for s in sensors:
            df[f"{s}_roll_mean_{w}"] = grouped[s].transform(
                lambda x: x.rolling(w, min_periods=1).mean()
            )
            df[f"{s}_roll_std_{w}"] = grouped[s].transform(
                lambda x: x.rolling(w, min_periods=1).std().fillna(0.0)
            )

    sw = config.SLOPE_WINDOW
    for s in sensors:
        df[f"{s}_slope_{sw}"] = grouped[s].transform(
            lambda x: _rolling_slope(x, sw)
        )
    # backfill slope NaNs (first sw-1 cycles of each unit) with 0 —
    # not enough history yet to estimate a trend, treat as "no trend detected"
    slope_cols = [f"{s}_slope_{sw}" for s in sensors]
    df[slope_cols] = df[slope_cols].fillna(0.0)

    return df


def drop_warmup_cycles(df, min_cycles=None):
    """Drop each unit's first (min_cycles - 1) cycles.

    Those rows cannot have a full-width rolling/slope window, so their
    std/slope features collapse to structural zeros — which the model reads
    as "flat, non-degrading engine". Training on them teaches the model a
    spurious mapping that the API can never reproduce (the API refuses
    histories shorter than MIN_HISTORY_CYCLES). Dropping them keeps the
    training feature distribution identical to the serving one.
    """
    if min_cycles is None:
        min_cycles = config.MIN_HISTORY_CYCLES
    return df[df["cycle"] >= min_cycles].reset_index(drop=True)


def get_feature_columns(df):
    """All model input columns: kept raw sensors + settings + engineered features."""
    sensors = _kept_sensor_cols()
    engineered = []
    for w in config.ROLLING_WINDOWS:
        for s in sensors:
            engineered += [f"{s}_roll_mean_{w}", f"{s}_roll_std_{w}"]
    engineered += [f"{s}_slope_{config.SLOPE_WINDOW}" for s in sensors]
    return config.SETTING_COLS + sensors + engineered
