"""Shared inference: load trained artifacts, engineer features for a request
payload, and produce RUL + risk predictions."""
import numpy as np
import pandas as pd
import joblib

from src.pipeline import config
from src.pipeline.features import add_features
from src.pipeline import ood

_ARTIFACTS = {}

REGRESSOR_KEY = {
    "Linear Regression": "linear_regressor",
    "Random Forest": "rf_regressor",
    "XGBoost": "xgb_regressor",
}
CLASSIFIER_KEY = {
    "Random Forest": "rf_classifier",
    "XGBoost": "xgb_classifier",
}


def load_artifacts():
    """Load trained models + preprocessing artifacts once, cached in memory."""
    if _ARTIFACTS:
        return _ARTIFACTS
    missing = [k for k, p in config.MODEL_FILES.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing trained artifacts: {missing}. Run "
            f"`python src/models/train_regression.py` then "
            f"`python src/models/train_classification.py` first."
        )
    for key in config.MODEL_FILES:
        _ARTIFACTS[key] = joblib.load(config.MODEL_FILES[key])
    return _ARTIFACTS


def get_regressor(artifacts):
    name = artifacts["best_regressor_name"]
    return name, artifacts[REGRESSOR_KEY[name]], name == "Linear Regression"


def get_classifier(artifacts):
    name = artifacts["best_classifier_name"]
    return name, artifacts[CLASSIFIER_KEY[name]]


def risk_band(predicted_rul):
    if predicted_rul < config.RISK_HIGH_RUL:
        return "high"
    if predicted_rul < config.RISK_MEDIUM_RUL:
        return "medium"
    return "low"


def readings_to_dataframe(readings, unit_label="0"):
    """Convert validated reading dicts into the raw-column frame the feature
    pipeline expects, tagged with a unit id so rolling windows group correctly."""
    rows = []
    for r in readings:
        row = {"unit": unit_label, "cycle": r["cycle"]}
        for i in (1, 2, 3):
            row[f"setting_{i}"] = r[f"setting_{i}"]
        for i in range(1, 22):
            row[f"sensor_{i}"] = r[f"sensor_{i}"]
        rows.append(row)
    return pd.DataFrame(rows)


def predict_from_readings(readings):
    """readings: list of validated dicts, oldest -> newest.

    Returns the prediction for the LATEST cycle in the supplied history.
    Raises ValueError (-> HTTP 400) if the history is too short to compute
    full-width features, or if the sensor combination is implausible.
    """
    if len(readings) < config.MIN_HISTORY_CYCLES:
        raise ValueError(
            f"Insufficient history: {len(readings)} cycle(s) supplied, but "
            f"{config.MIN_HISTORY_CYCLES} are required. Rolling-std and "
            f"degradation-slope features need a full {config.MIN_HISTORY_CYCLES}-cycle "
            f"window; with less history they collapse to zero, which the model "
            f"reads as a healthy, non-degrading engine and causes it to "
            f"significantly OVER-predict remaining life. Submit at least "
            f"{config.MIN_HISTORY_CYCLES} consecutive cycles."
        )

    artifacts = load_artifacts()
    df = readings_to_dataframe(readings)

    # Reject physically impossible sensor combinations that pass per-field
    # range checks (per-field bounds cannot see correlations between sensors).
    guard = artifacts["ood_guard"]
    distance = float(ood.score(guard, df.tail(1))[0])
    if distance > guard["reject_threshold"]:
        raise ValueError(
            f"Implausible sensor combination: Mahalanobis distance {distance:.1f} "
            f"from the training distribution exceeds the reject threshold "
            f"{guard['reject_threshold']:.1f}. Individual sensor values are "
            f"in range, but their combination does not correspond to any "
            f"physically observed engine state — check for swapped, stale, or "
            f"unit-mismatched sensor channels."
        )

    df_feat = add_features(df)
    feature_cols = artifacts["feature_columns"]
    latest = df_feat.iloc[[-1]][feature_cols]

    reg_name, reg_model, needs_scaling = get_regressor(artifacts)
    X = artifacts["scaler"].transform(latest) if needs_scaling else latest
    predicted_rul = float(np.clip(reg_model.predict(X)[0], 0, None))

    clf_name, clf_model = get_classifier(artifacts)
    proba = clf_model.predict_proba(latest)[0]
    maintenance_prob = float(proba[1])

    return {
        "predicted_rul": round(predicted_rul, 2),
        "risk_level": risk_band(predicted_rul),
        "needs_maintenance": bool(maintenance_prob >= config.DECISION_THRESHOLD),
        "maintenance_probability": round(maintenance_prob, 4),
        "confidence": round(float(max(proba)), 4),
        "decision_threshold": config.DECISION_THRESHOLD,
        "input_plausibility_warning": distance > guard["warn_threshold"],
        "cycles_supplied": len(readings),
        "model_used_regressor": reg_name,
        "model_used_classifier": clf_name,
        "horizon_cycles": config.MAINTENANCE_HORIZON,
    }
