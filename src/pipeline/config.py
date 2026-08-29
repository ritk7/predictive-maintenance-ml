"""
Central configuration for the predictive maintenance pipeline.
All tunable thresholds, paths, and hyperparameters live here — nothing
should be hardcoded inline in pipeline/model/api code.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models_saved"

TRAIN_FILE = DATA_DIR / "train_FD001.txt"
TEST_FILE = DATA_DIR / "test_FD001.txt"
RUL_FILE = DATA_DIR / "RUL_FD001.txt"

# --- Raw column schema ---------------------------------------------------
# C-MAPSS files are whitespace-delimited with no header:
# unit, cycle, 3 op settings, 21 sensors
INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SETTING_COLS + SENSOR_COLS

# --- Sensors dropped for near-zero variance -----------------------------
# Determined empirically in eda.py (std/mean check on training data) and
# confirmed against published C-MAPSS literature: under FD001's single
# operating condition, these sensors are physically pinned constant
# (e.g. fixed altitude/Mach/throttle-resolver-angle related channels,
# or sensors that saturate) and carry no degradation signal.
DROP_SENSORS = ["sensor_1", "sensor_5", "sensor_6", "sensor_10",
                 "sensor_16", "sensor_18", "sensor_19"]

# --- Feature engineering -------------------------------------------------
ROLLING_WINDOWS = [5, 10]     # rolling mean/std window sizes (cycles)
SLOPE_WINDOW = 10             # window for degradation-trend slope feature

# Minimum cycle history required to compute every engineered feature at full
# width. Below this, rolling std/slope collapse to structural zeros, which the
# models read as "flat, non-degrading engine" and therefore over-predict RUL.
# This is enforced BOTH at training time (degenerate warm-up rows are dropped)
# and at inference time (the API rejects short histories) so the feature
# distribution the model is served matches the one it was trained on.
MIN_HISTORY_CYCLES = max(ROLLING_WINDOWS + [SLOPE_WINDOW])

# --- RUL clipping ----------------------------------------------------------
# Published C-MAPSS approaches (Heimes 2008, Saxena et al.) clip RUL at a
# ceiling because early-life degradation is not observable in the sensors
# (engines run "healthy" for a long flat stretch before degrading) — trying
# to regress the exact RUL during that flat stretch just adds noise the
# model cannot learn from. Clipping turns the target into "healthy vs.
# degrading-on-a-known-trajectory," which is what the sensors can actually
# support.
RUL_CLIP = 125

# --- Classification ------------------------------------------------------
# "Needs maintenance within N cycles". We use N=30 (rather than 20):
# FD001 engines degrade over ~150-360 cycles, and a 30-cycle horizon gives
# maintenance planners realistic lead time to schedule downtime/parts
# without flagging so early that the alert is ignored. Empirically N=20
# leaves only 0.94% positives in the held-out test set (too sparse to
# estimate recall stably), while N=30 gives 2.54%; see README §6.
MAINTENANCE_HORIZON = 30

# Decision threshold on predicted P(needs maintenance).
# NOT 0.5: in predictive maintenance a false negative (engine fails in
# service) costs far more than a false positive (an unnecessary inspection),
# so the operating point is tuned on the recall-weighted F-beta score
# (beta=2) rather than left at the default. Selected empirically in
# train_classification.py's threshold sweep; see README §6.
DECISION_THRESHOLD = 0.30

# beta for F-beta model selection. beta=2 weights recall 2x precision.
FBETA = 2.0

# --- Train/validation split (by unit, never by row) -----------------------
VAL_FRACTION = 0.2
RANDOM_STATE = 42

# --- Model hyperparameters -------------------------------------------------
RF_REGRESSOR_PARAMS = dict(
    n_estimators=200, max_depth=12, min_samples_leaf=5,
    random_state=RANDOM_STATE, n_jobs=-1,
)
XGB_REGRESSOR_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1,
)
RF_CLASSIFIER_PARAMS = dict(
    n_estimators=200, max_depth=12, min_samples_leaf=5,
    class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
)
XGB_CLASSIFIER_PARAMS = dict(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, n_jobs=-1,
)

# --- API / dashboard risk thresholds (predicted RUL -> risk band) --------
RISK_HIGH_RUL = 20    # predicted RUL below this => "high" risk
RISK_MEDIUM_RUL = 50  # predicted RUL below this (and >= high) => "medium"
# else => "low"

# --- Saved model file names -----------------------------------------------
MODEL_FILES = {
    "rf_regressor": MODELS_DIR / "rf_regressor.joblib",
    "xgb_regressor": MODELS_DIR / "xgb_regressor.joblib",
    "linear_regressor": MODELS_DIR / "linear_regressor.joblib",
    "rf_classifier": MODELS_DIR / "rf_classifier.joblib",
    "xgb_classifier": MODELS_DIR / "xgb_classifier.joblib",
    "scaler": MODELS_DIR / "scaler.joblib",
    "feature_columns": MODELS_DIR / "feature_columns.joblib",
    "best_regressor_name": MODELS_DIR / "best_regressor_name.joblib",
    "best_classifier_name": MODELS_DIR / "best_classifier_name.joblib",
    "ood_guard": MODELS_DIR / "ood_guard.joblib",
}

# --- Out-of-distribution input guard --------------------------------------
# Per-field range checks accept physically impossible sensor *combinations*
# (e.g. every sensor simultaneously at its individual minimum). A Mahalanobis
# distance against the training sensor distribution catches those. Distances
# are calibrated on training data; see src/pipeline/ood.py.
OOD_WARN_PERCENTILE = 99.9   # above this training percentile -> warn in response
OOD_REJECT_FACTOR = 3.0      # above WARN_THRESHOLD * this -> hard 400 reject
