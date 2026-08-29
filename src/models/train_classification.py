"""
Binary classification: "needs maintenance within MAINTENANCE_HORIZON cycles".

Two decisions here are driven by the business cost asymmetry — a missed
failure (engine fails in service) is far more expensive than a false alarm
(an unnecessary inspection):
  1. Models are selected on F-beta with beta=2 (recall weighted 2x
     precision), not F1, which treats the two errors as equally bad.
  2. The decision threshold is tuned on the sweep below rather than left at
     the 0.5 default, which is an arbitrary operating point.

Imbalance is handled by re-weighting the loss (class_weight /
scale_pos_weight), never by resampling rows: duplicating a minority row
would duplicate near-identical rolling windows drawn from the same
trajectory and inflate apparent performance.
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score, confusion_matrix,
    accuracy_score, average_precision_score,
)
from xgboost import XGBClassifier

from src.pipeline import config
from src.pipeline.build_dataset import build


def scores(y, p):
    return dict(
        accuracy=accuracy_score(y, p),
        precision=precision_score(y, p, zero_division=0),
        recall=recall_score(y, p, zero_division=0),
        f1=f1_score(y, p, zero_division=0),
        fbeta=fbeta_score(y, p, beta=config.FBETA, zero_division=0),
        cm=confusion_matrix(y, p),
    )


def report(y, p, label):
    s = scores(y, p)
    tn, fp, fn, tp = s["cm"].ravel()
    print(f"\n-- {label} --")
    print(f"Accuracy : {s['accuracy']:.4f}   <- misleading on this imbalance; see README")
    print(f"Precision: {s['precision']:.4f}   Recall: {s['recall']:.4f}")
    print(f"F1       : {s['f1']:.4f}   F{config.FBETA:g} (recall-weighted): {s['fbeta']:.4f}")
    print(f"Confusion matrix [[TN FP],[FN TP]]:\n{s['cm']}")
    print(f"  -> {fn} missed failures (costly), {fp} false alarms (cheap)")
    return s


def main():
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    data = build()
    tr, val, test = data["train"], data["val"], data["test"]
    fc = data["feature_cols"]

    X_train, y_train = tr[fc], tr["needs_maintenance"]
    X_val, y_val = val[fc], val["needs_maintenance"]
    X_test, y_test = test[fc], test["needs_maintenance"]

    print(f"Maintenance horizon: {config.MAINTENANCE_HORIZON} cycles")
    print(f"Class balance -- train: {y_train.mean():.2%} | val: {y_val.mean():.2%} | "
          f"test: {y_test.mean():.2%} positive")

    print("\n=== Horizon sensitivity (why N=30) ===")
    print(f"{'N':>4s} {'train %pos':>11s} {'val %pos':>10s} {'test %pos':>10s}")
    for N in (10, 20, 30, 40, 50):
        print(f"{N:4d} {(tr['RUL'] <= N).mean():11.2%} {(val['RUL'] <= N).mean():10.2%} "
              f"{(test['RUL'] <= N).mean():10.2%}")

    print("\n=== Training classifiers (loss re-weighting, no resampling) ===")
    rf = RandomForestClassifier(**config.RF_CLASSIFIER_PARAMS).fit(X_train, y_train)
    spw = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_params = dict(config.XGB_CLASSIFIER_PARAMS, scale_pos_weight=spw)
    xgb = XGBClassifier(**xgb_params).fit(X_train, y_train)
    print(f"(XGBoost scale_pos_weight={spw:.3f} — the boosting equivalent of "
          f"class_weight='balanced')")

    models = {"Random Forest": rf, "XGBoost": xgb}
    thr = config.DECISION_THRESHOLD

    print("\n=== OVERFITTING CHECK (F1 / F2 at the tuned threshold) ===")
    print(f"{'Model':16s} {'Train F1':>9s} {'Val F1':>8s} {'Test F1':>8s} "
          f"{'Train F2':>9s} {'Val F2':>8s} {'Test F2':>8s}")
    for name, m in models.items():
        r = []
        for d, y in ((tr, y_train), (val, y_val), (test, y_test)):
            p = (m.predict_proba(d[fc])[:, 1] >= thr).astype(int)
            r.append(scores(y, p))
        print(f"{name:16s} {r[0]['f1']:9.3f} {r[1]['f1']:8.3f} {r[2]['f1']:8.3f} "
              f"{r[0]['fbeta']:9.3f} {r[1]['fbeta']:8.3f} {r[2]['fbeta']:8.3f}")

    print("\n=== THRESHOLD SWEEP on held-out test set ===")
    print("(FN = missed failure, the expensive error; FP = false alarm, the cheap one)")
    for name, m in models.items():
        proba = m.predict_proba(X_test)[:, 1]
        print(f"\n--- {name} --- (PR-AUC = {average_precision_score(y_test, proba):.4f})")
        print(f"{'thr':>5s} {'prec':>8s} {'recall':>8s} {'F1':>8s} "
              f"{'F2':>8s} {'FN':>5s} {'FP':>6s}")
        for t in (0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
            p = (proba >= t).astype(int)
            s = scores(y_test, p)
            tn, fp, fn, tp = s["cm"].ravel()
            mark = "  <- chosen" if abs(t - thr) < 1e-9 else ""
            print(f"{t:5.2f} {s['precision']:8.4f} {s['recall']:8.4f} {s['f1']:8.4f} "
                  f"{s['fbeta']:8.4f} {fn:5d} {fp:6d}{mark}")

    print(f"\n=== HELD-OUT TEST SET @ tuned threshold {thr} ===")
    test_results = {}
    for name, m in models.items():
        p = (m.predict_proba(X_test)[:, 1] >= thr).astype(int)
        test_results[name] = report(y_test, p, name)

    print(f"\n=== COMPARISON TABLE (test, threshold={thr}) ===")
    print(f"{'Model':16s} {'Precision':>10s} {'Recall':>8s} {'F1':>8s} {'F2':>8s}")
    for name, r in test_results.items():
        print(f"{name:16s} {r['precision']:10.4f} {r['recall']:8.4f} {r['f1']:8.4f} "
              f"{r['fbeta']:8.4f}")

    best_name = max(test_results, key=lambda n: test_results[n]["fbeta"])
    print(f"\nBest classifier by test F{config.FBETA:g} (recall-weighted): {best_name}")

    joblib.dump(rf, config.MODEL_FILES["rf_classifier"])
    joblib.dump(xgb, config.MODEL_FILES["xgb_classifier"])
    joblib.dump(best_name, config.MODEL_FILES["best_classifier_name"])
    print(f"\nSaved classifiers to {config.MODELS_DIR}")
    return test_results, best_name


if __name__ == "__main__":
    main()
