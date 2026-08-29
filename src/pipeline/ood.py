"""Out-of-distribution input guard.

Per-field range validation accepts physically impossible sensor
*combinations* — e.g. all 21 sensors simultaneously pinned to their
individual minima, which is thermodynamically nonsense but passes every
individual bound. This module fits a Mahalanobis distance model over the
kept raw sensors on training data so such inputs can be detected.

Thresholds are calibrated from the training distances themselves rather
than assumed, because the sensors are strongly correlated and the
chi-square approximation would be optimistic.
"""
import numpy as np

from src.pipeline import config


def _kept_sensors():
    return [c for c in config.SENSOR_COLS if c not in config.DROP_SENSORS]


def fit_ood_guard(train_df):
    """Fit mean/inverse-covariance on training sensors and calibrate cutoffs."""
    cols = _kept_sensors()
    X = train_df[cols].to_numpy(dtype=float)
    mean = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    # pinv is used rather than inv: some C-MAPSS sensors are near-collinear,
    # which makes the covariance matrix ill-conditioned.
    inv_cov = np.linalg.pinv(cov)

    d = _mahalanobis(X, mean, inv_cov)
    warn_threshold = float(np.percentile(d, config.OOD_WARN_PERCENTILE))
    return {
        "columns": cols,
        "mean": mean,
        "inv_cov": inv_cov,
        "warn_threshold": warn_threshold,
        "reject_threshold": warn_threshold * config.OOD_REJECT_FACTOR,
    }


def _mahalanobis(X, mean, inv_cov):
    delta = X - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, inv_cov, delta))


def score(guard, df):
    """Mahalanobis distance for each row of df (must contain sensor columns)."""
    X = df[guard["columns"]].to_numpy(dtype=float)
    return _mahalanobis(X, guard["mean"], guard["inv_cov"])
