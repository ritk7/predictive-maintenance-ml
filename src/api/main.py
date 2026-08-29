"""FastAPI predictive-maintenance API.

POST /predict                  RUL + risk for a submitted engine history.
GET  /engines                  risk status for all test-set engines.
GET  /engines/{id}/history     sensor cycle history for one test engine.
GET  /health                   liveness check.
"""
from contextlib import asynccontextmanager
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from src.pipeline import config
from src.pipeline.build_dataset import build
from src.api.schemas import (
    PredictRequest, PredictResponse, EngineListResponse, EngineStatus,
)
from src.api.inference import (
    load_artifacts, predict_from_readings, get_regressor, get_classifier, risk_band,
)


@lru_cache(maxsize=1)
def get_dataset():
    """Feature engineering over the full dataset takes ~9s; the result is
    immutable for the process lifetime, so it is built once and cached.
    Previously this ran per-request, making /engines/{id}/history a ~9s call
    and a trivial denial-of-service vector."""
    return build()


_ENGINE_SNAPSHOT = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_artifacts()
        get_dataset()
        _build_engine_snapshot()
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
    yield


app = FastAPI(
    title="Turbofan Predictive Maintenance API",
    description="Predicts Remaining Useful Life (RUL) and maintenance risk for "
                "jet engines using NASA C-MAPSS FD001 sensor data.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return schema violations as 400s with field-level detail (not 422)."""
    errors = [
        {"field": ".".join(str(p) for p in e["loc"]), "message": e["msg"]}
        for e in exc.errors()
    ]
    return JSONResponse(
        status_code=400,
        content={"detail": "Invalid request payload.", "errors": errors},
    )


def _build_engine_snapshot():
    global _ENGINE_SNAPSHOT
    try:
        artifacts = load_artifacts()
    except FileNotFoundError:
        _ENGINE_SNAPSHOT = []
        return

    data = get_dataset()
    test, feature_cols = data["test"], data["feature_cols"]

    reg_name, reg_model, needs_scaling = get_regressor(artifacts)
    clf_name, clf_model = get_classifier(artifacts)

    latest = test.groupby("unit").tail(1).sort_values("unit")
    X = latest[feature_cols]
    X_reg = artifacts["scaler"].transform(X) if needs_scaling else X
    preds = np.clip(reg_model.predict(X_reg), 0, None)
    proba = clf_model.predict_proba(X)[:, 1]

    _ENGINE_SNAPSHOT = [
        {
            "unit_id": str(int(row["unit"])),
            "last_cycle": int(row["cycle"]),
            "predicted_rul": round(float(pred), 2),
            "risk_level": risk_band(pred),
            "needs_maintenance": bool(prob >= config.DECISION_THRESHOLD),
            "maintenance_probability": round(float(prob), 4),
        }
        for (_, row), pred, prob in zip(latest.iterrows(), preds, proba)
    ]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = predict_from_readings([r.model_dump() for r in req.readings])
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return PredictResponse(unit_id=req.unit_id, **result)


@app.get("/engines", response_model=EngineListResponse)
def list_engines():
    if _ENGINE_SNAPSHOT is None:
        _build_engine_snapshot()
    if not _ENGINE_SNAPSHOT:
        raise HTTPException(
            status_code=503,
            detail="No trained models available. Train the models first.",
        )
    return EngineListResponse(
        engines=[EngineStatus(**e) for e in _ENGINE_SNAPSHOT],
        count=len(_ENGINE_SNAPSHOT),
        risk_thresholds={
            "high_below_rul": config.RISK_HIGH_RUL,
            "medium_below_rul": config.RISK_MEDIUM_RUL,
            "decision_threshold": config.DECISION_THRESHOLD,
        },
    )


@app.get("/engines/{unit_id}/history")
def engine_history(unit_id: str):
    """Sensor cycle history for one test engine, for dashboard trend charts."""
    try:
        uid = int(unit_id)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unit_id must be numeric, got '{unit_id}'.")

    test = get_dataset()["test"]
    sub = test[test["unit"] == uid]
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"No engine with unit_id '{unit_id}' in test set.")

    sensor_cols = [c for c in config.SENSOR_COLS if c not in config.DROP_SENSORS]
    return {"unit_id": unit_id, "history": sub[["cycle"] + sensor_cols].to_dict(orient="list")}
