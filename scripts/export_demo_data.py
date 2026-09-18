"""Export real model output as a static JSON payload for the hosted demo.

The demo page is served from GitHub Pages and cannot reach a local API, so
everything it shows is precomputed here from the actual trained models and
the real C-MAPSS test set — predictions, ground truth, sensor traces, the
threshold sweep, and the model comparison tables. No synthetic numbers.

Usage:  PYTHONPATH=. python scripts/export_demo_data.py
"""
import json

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score, confusion_matrix, mean_absolute_error,
    mean_squared_error, precision_score, recall_score, f1_score, fbeta_score,
)

from src.pipeline import config
from src.pipeline.build_dataset import build
from src.api.inference import risk_band

OUT_PATH = config.PROJECT_ROOT / "docs" / "demo_data.json"
# Sensors surfaced in the demo's trend charts, with human-readable names.
# sensor_4 is the headline one: LPT outlet temperature rises as the high
# pressure compressor degrades, which is the physical signal the models key on.
TRACE_SENSORS = {
    "sensor_4": "LPT outlet temperature (°R)",
    "sensor_11": "HPC static pressure (psia)",
    "sensor_9": "Physical core speed (rpm)",
    "sensor_15": "Bypass ratio",
}


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def main():
    data = build()
    tr, val, test = data["train"], data["val"], data["test"]
    fc = data["feature_cols"]

    art = {k: joblib.load(p) for k, p in config.MODEL_FILES.items()}
    scaler = art["scaler"]
    reg_name = art["best_regressor_name"]
    clf_name = art["best_classifier_name"]

    regressors = {
        "Linear Regression": (art["linear_regressor"], True),
        "Random Forest": (art["rf_regressor"], False),
        "XGBoost": (art["xgb_regressor"], False),
    }
    classifiers = {"Random Forest": art["rf_classifier"], "XGBoost": art["xgb_classifier"]}

    reg_model, reg_scaled = regressors[reg_name]
    clf_model = classifiers[clf_name]

    # ---------- fleet: one row per test engine, at its latest cycle ----------
    latest = test.groupby("unit").tail(1).sort_values("unit")
    X = latest[fc]
    X_reg = scaler.transform(X) if reg_scaled else X
    preds = np.clip(reg_model.predict(X_reg), 0, None)
    probs = clf_model.predict_proba(X)[:, 1]

    fleet = []
    for (_, row), pred, prob in zip(latest.iterrows(), preds, probs):
        unit = int(row["unit"])
        fleet.append({
            "unit": unit,
            "last_cycle": int(row["cycle"]),
            "predicted_rul": round(float(pred), 1),
            "true_rul": int(row["RUL"]),
            "risk": risk_band(pred),
            "probability": round(float(prob), 4),
            "flagged": bool(prob >= config.DECISION_THRESHOLD),
        })

    # ---------- sensor traces per engine (subsampled to keep payload small) ----------
    traces = {}
    for unit, grp in test.groupby("unit"):
        grp = grp.sort_values("cycle")
        step = max(1, len(grp) // 60)          # <= ~60 points per engine
        sub = grp.iloc[::step]
        traces[str(int(unit))] = {
            "cycle": [int(c) for c in sub["cycle"]],
            **{s: [round(float(v), 2) for v in sub[s]] for s in TRACE_SENSORS},
        }

    # ---------- real threshold sweep, fine-grained ----------
    y_test = test["needs_maintenance"].to_numpy()
    sweeps = {}
    for name, model in classifiers.items():
        p = model.predict_proba(test[fc])[:, 1]
        points = []
        for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
            pred = (p >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
            points.append({
                "t": float(t),
                "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
                "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
                "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
                "f2": round(float(fbeta_score(y_test, pred, beta=2, zero_division=0)), 4),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            })
        sweeps[name] = {
            "pr_auc": round(float(average_precision_score(y_test, p)), 4),
            "points": points,
        }

    # ---------- regression comparison, both protocols ----------
    reg_table = []
    for name, (m, scaled) in regressors.items():
        def pred_on(df):
            Xd = df[fc]
            return np.clip(m.predict(scaler.transform(Xd) if scaled else Xd), 0, None)
        p_all, p_last = pred_on(test), pred_on(latest)
        reg_table.append({
            "model": name,
            "rmse_all": round(rmse(test["RUL_clipped"], p_all), 3),
            "mae_all": round(float(mean_absolute_error(test["RUL_clipped"], p_all)), 3),
            "rmse_last": round(rmse(latest["RUL_clipped"], p_last), 3),
            "mae_last": round(float(mean_absolute_error(latest["RUL_clipped"], p_last)), 3),
            "rmse_train": round(rmse(tr["RUL_clipped"], pred_on(tr)), 3),
            "rmse_val": round(rmse(val["RUL_clipped"], pred_on(val)), 3),
            "selected": name == reg_name,
        })

    payload = {
        "meta": {
            "dataset": "NASA C-MAPSS FD001",
            "train_engines": int(tr["unit"].nunique() + val["unit"].nunique()),
            "test_engines": int(test["unit"].nunique()),
            "train_rows": int(len(tr) + len(val)),
            "test_rows": int(len(test)),
            "n_features": len(fc),
            "dropped_sensors": config.DROP_SENSORS,
            "rul_clip": config.RUL_CLIP,
            "horizon": config.MAINTENANCE_HORIZON,
            "decision_threshold": config.DECISION_THRESHOLD,
            "min_history": config.MIN_HISTORY_CYCLES,
            "risk_high_below": config.RISK_HIGH_RUL,
            "risk_medium_below": config.RISK_MEDIUM_RUL,
            "selected_regressor": reg_name,
            "selected_classifier": clf_name,
            "positive_rate_test": round(float(y_test.mean()), 4),
        },
        "sensor_labels": TRACE_SENSORS,
        "fleet": fleet,
        "traces": traces,
        "sweeps": sweeps,
        "regression": reg_table,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, separators=(",", ":")))
    kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {OUT_PATH} ({kb:.0f} KB)")
    print(f"  fleet engines : {len(fleet)}")
    print(f"  traces        : {len(traces)} engines x {len(TRACE_SENSORS)} sensors")
    print(f"  sweep points  : {len(sweeps[clf_name]['points'])} per classifier")
    high = sum(1 for f in fleet if f["risk"] == "high")
    med = sum(1 for f in fleet if f["risk"] == "medium")
    print(f"  risk split    : {high} high / {med} medium / {len(fleet)-high-med} low")


if __name__ == "__main__":
    main()
