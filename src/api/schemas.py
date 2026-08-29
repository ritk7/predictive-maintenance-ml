"""Pydantic request/response models with explicit validation ranges."""
import math
from typing import List

from pydantic import BaseModel, Field, field_validator

from src.pipeline import config

N_SETTINGS = 3
N_SENSORS = 21
MAX_READINGS = 500

# Physically plausible ranges observed in C-MAPSS FD001 training data,
# widened slightly as guard rails (not tight bounds) to catch obviously
# malformed input (wrong units, swapped fields, sensor stuck at 0, etc.)
# without rejecting legitimate degraded-engine readings.
SENSOR_RANGES = {
    1: (400, 700), 2: (500, 700), 3: (1300, 1650), 4: (1050, 1550),
    5: (10, 20), 6: (15, 25), 7: (500, 650), 8: (2000, 2500),
    9: (8000, 9500), 10: (0.5, 2.0), 11: (35, 60), 12: (450, 600),
    13: (2000, 2500), 14: (7500, 8500), 15: (7, 10), 16: (0.01, 0.1),
    17: (300, 450), 18: (2000, 2500), 19: (80, 120), 20: (30, 45),
    21: (18, 28),
}
SETTING_RANGES = {1: (-0.1, 45), 2: (-0.1, 1.0), 3: (60, 110)}


class CycleReading(BaseModel):
    cycle: int = Field(..., ge=1, description="Time cycle index for this reading")
    setting_1: float
    setting_2: float
    setting_3: float
    sensor_1: float
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_5: float
    sensor_6: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_10: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_16: float
    sensor_17: float
    sensor_18: float
    sensor_19: float
    sensor_20: float
    sensor_21: float

    @field_validator("*")
    @classmethod
    def no_nan_inf(cls, v, info):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            raise ValueError(f"{info.field_name} must be a finite number, got {v}")
        return v


class PredictRequest(BaseModel):
    unit_id: str = Field(..., min_length=1, description="Engine identifier")
    readings: List[CycleReading] = Field(
        ..., description="Recent cycle history, ordered oldest -> newest"
    )

    @field_validator("readings")
    @classmethod
    def validate_readings(cls, v):
        if len(v) < config.MIN_HISTORY_CYCLES:
            raise ValueError(
                f"Insufficient history: {len(v)} cycle(s) supplied, but "
                f"{config.MIN_HISTORY_CYCLES} are required. Rolling-std and "
                f"degradation-slope features need a full "
                f"{config.MIN_HISTORY_CYCLES}-cycle window; with less history they "
                f"collapse to zero, which the model reads as a healthy, "
                f"non-degrading engine and causes it to significantly OVER-predict "
                f"remaining life."
            )
        if len(v) > MAX_READINGS:
            raise ValueError(f"Too many readings in one request (max {MAX_READINGS}).")
        cycles = [r.cycle for r in v]
        if cycles != sorted(cycles):
            raise ValueError("readings must be ordered by ascending cycle.")
        if len(set(cycles)) != len(cycles):
            raise ValueError("duplicate cycle values found in readings.")
        for r in v:
            for i in (1, 2, 3):
                lo, hi = SETTING_RANGES[i]
                val = getattr(r, f"setting_{i}")
                if not (lo <= val <= hi):
                    raise ValueError(
                        f"setting_{i}={val} out of plausible range [{lo}, {hi}] "
                        f"at cycle {r.cycle}."
                    )
            for i in range(1, N_SENSORS + 1):
                lo, hi = SENSOR_RANGES[i]
                val = getattr(r, f"sensor_{i}")
                if not (lo <= val <= hi):
                    raise ValueError(
                        f"sensor_{i}={val} out of plausible range [{lo}, {hi}] "
                        f"at cycle {r.cycle}. If this is a genuine extreme reading, "
                        f"verify the sensor/unit conversion before resubmitting."
                    )
        return v


class PredictResponse(BaseModel):
    unit_id: str
    predicted_rul: float
    risk_level: str
    needs_maintenance: bool
    maintenance_probability: float
    confidence: float
    decision_threshold: float
    input_plausibility_warning: bool
    cycles_supplied: int
    model_used_regressor: str
    model_used_classifier: str
    horizon_cycles: int


class EngineStatus(BaseModel):
    unit_id: str
    last_cycle: int
    predicted_rul: float
    risk_level: str
    needs_maintenance: bool
    maintenance_probability: float


class EngineListResponse(BaseModel):
    engines: List[EngineStatus]
    count: int
    risk_thresholds: dict
