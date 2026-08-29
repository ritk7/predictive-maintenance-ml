"""
Train and compare 3 RUL regressors: Linear Regression baseline, Random
Forest, XGBoost.

Reports metrics under BOTH evaluation protocols, because they differ a lot
and quoting only the flattering one would be dishonest:
  * all-rows      — every test engine-cycle (13k rows)
  * last-cycle    — one row per test engine (100 rows), the protocol used by
                    published C-MAPSS FD001 benchmarks
and against both the clipped and the raw (unclipped) RUL target.
"""
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor

from src.pipeline import config
from src.pipeline.build_dataset import build
from src.pipeline.ood import fit_ood_guard


def rmse_mae(y_true, y_pred):
    return (float(np.sqrt(mean_squared_error(y_true, y_pred))),
            float(mean_absolute_error(y_true, y_pred)))


def main():
    config.MODELS_DIR.mkdir(exist_ok=True, parents=True)
    data = build()
    tr, val, test = data["train"], data["val"], data["test"]
    feature_cols = data["feature_cols"]

    X_train, y_train = tr[feature_cols], tr["RUL_clipped"]
    X_val, y_val = val[feature_cols], val["RUL_clipped"]
    X_test, y_test = test[feature_cols], test["RUL_clipped"]

    scaler = StandardScaler().fit(X_train)   # fit on TRAIN ONLY

    def design(df, scaled):
        return scaler.transform(df[feature_cols]) if scaled else df[feature_cols]

    print("\n=== Training models ===")
    models = {
        "Linear Regression": (LinearRegression().fit(scaler.transform(X_train), y_train), True),
        "Random Forest": (RandomForestRegressor(**config.RF_REGRESSOR_PARAMS).fit(X_train, y_train), False),
        "XGBoost": (XGBRegressor(**config.XGB_REGRESSOR_PARAMS).fit(X_train, y_train), False),
    }

    def predict(model, scaled, df):
        return np.clip(model.predict(design(df, scaled)), 0, None)

    # ---- overfitting check: train vs val vs test, like-for-like ----
    print("\n=== OVERFITTING CHECK (RMSE on clipped target, all rows) ===")
    print(f"{'Model':20s} {'Train':>9s} {'Val':>9s} {'Test':>9s} {'Val-Train gap':>14s}")
    for name, (m, s) in models.items():
        rtr = rmse_mae(y_train, predict(m, s, tr))[0]
        rv = rmse_mae(y_val, predict(m, s, val))[0]
        rte = rmse_mae(y_test, predict(m, s, test))[0]
        print(f"{name:20s} {rtr:9.3f} {rv:9.3f} {rte:9.3f} {rv - rtr:14.3f}")

    # ---- headline comparison table ----
    print("\n=== COMPARISON TABLE — held-out test set ===")
    print(f"{'Model':20s} {'RMSE(all)':>10s} {'MAE(all)':>9s} "
          f"{'RMSE(last)':>11s} {'MAE(last)':>10s} {'RMSE(last,raw)':>15s}")
    last = test.groupby("unit").tail(1)
    test_scores = {}
    for name, (m, s) in models.items():
        p_all = predict(m, s, test)
        p_last = predict(m, s, last)
        r_all, m_all = rmse_mae(y_test, p_all)
        r_last, m_last = rmse_mae(last["RUL_clipped"], p_last)
        r_last_raw, _ = rmse_mae(last["RUL"], p_last)
        test_scores[name] = dict(rmse_all=r_all, mae_all=m_all, rmse_last=r_last,
                                  mae_last=m_last, rmse_last_raw=r_last_raw)
        print(f"{name:20s} {r_all:10.3f} {m_all:9.3f} {r_last:11.3f} "
              f"{m_last:10.3f} {r_last_raw:15.3f}")
    print(f"\n(all = {len(test)} engine-cycles; last = {len(last)} engines, "
          f"one row each — the published-benchmark protocol)")

    # Model selection uses the last-cycle protocol: that is the decision the
    # system actually makes in production (score an engine as of *now*).
    best_name = min(test_scores, key=lambda n: test_scores[n]["rmse_last"])
    print(f"\nBest model by last-cycle test RMSE: {best_name}")

    # ---- per-engine sanity check, incl. deliberately hard cases ----
    print("\n=== Sanity check: predicted vs actual RUL, individual test engines ===")
    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_units = sorted(rng.choice(last["unit"].unique(), 5, replace=False).tolist())
    life = test.groupby("unit")["cycle"].max() + last.set_index("unit")["RUL"]
    hard_units = [int(life.idxmin()), int(life.idxmax())]
    for name, (m, s) in models.items():
        print(f"\n-- {name} --")
        for u in sample_units:
            row = last[last["unit"] == u]
            pred = predict(m, s, row)[0]
            print(f"  unit {u:3d}: predicted RUL={pred:7.2f}  |  actual RUL={row['RUL'].iloc[0]:6.0f}")
        for u, lbl in zip(hard_units, ["fastest-failing", "slowest-failing"]):
            row = last[last["unit"] == u]
            pred = predict(m, s, row)[0]
            print(f"  unit {u:3d}: predicted RUL={pred:7.2f}  |  actual RUL={row['RUL'].iloc[0]:6.0f}"
                  f"   <-- {lbl} engine in fleet")

    # ---- persist artifacts ----
    joblib.dump(models["Linear Regression"][0], config.MODEL_FILES["linear_regressor"])
    joblib.dump(models["Random Forest"][0], config.MODEL_FILES["rf_regressor"])
    joblib.dump(models["XGBoost"][0], config.MODEL_FILES["xgb_regressor"])
    joblib.dump(scaler, config.MODEL_FILES["scaler"])
    joblib.dump(feature_cols, config.MODEL_FILES["feature_columns"])
    joblib.dump(best_name, config.MODEL_FILES["best_regressor_name"])
    joblib.dump(fit_ood_guard(tr), config.MODEL_FILES["ood_guard"])
    print(f"\nSaved models + OOD guard to {config.MODELS_DIR}")
    return test_scores, best_name


if __name__ == "__main__":
    main()
