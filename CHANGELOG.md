# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Documentation

- Added an example 400 error response body (field-level validation detail)
  to the API section, so callers can see the actual shape of a rejected
  request alongside the existing success-response examples.
- Added an example response body for `GET /health` to the API section,
  completing example coverage for all four endpoints (previously only
  `/predict`, `/engines`, and `/engines/{id}/history` had one).
- Added a reference table of the key tunable constants in `config.py`
  (`MIN_HISTORY_CYCLES`, `DECISION_THRESHOLD`, the risk/OOD thresholds) to
  the API section, so callers don't need to read the source to know what
  values drive the API's behavior.
- Added example response bodies for `GET /engines` and
  `GET /engines/{unit_id}/history` to the API section, matching the
  existing `POST /predict` example.
- Added a sample `curl` request and response body for `POST /predict` to
  the API section, alongside the existing response-field table.
- Added a Troubleshooting section covering the most common setup errors
  (missing data files, missing `PYTHONPATH`, predicting before training,
  port conflicts, and the macOS OpenMP error).
- Documented every `/predict` response field in a table in the API section,
  so callers don't need to read `schemas.py` to know what comes back.
- Added a table of contents linking to all 12 README sections.
- Added Python version requirement (3.9+) to setup instructions.
- Documented `sample_request.py` and Makefile target equivalents in the
  quickstart section.
- Clarified that the `libomp` / OpenMP install note applies to macOS only.
- Added a summary Twitter card since the page ships no `og:image`.

## Frontend

- Fixed a runaway canvas resize and an opaque gradient fill in the hosted
  demo; made canvas sizing structurally incapable of unbounded growth.

## Initial release

- Added MIT license and fixed findings from a frontend audit.
- Added the hosted static demo console, project Makefile, and the demo
  build pipeline (`export_demo_data.py` + `build_demo.py`).
- Built the predictive maintenance system for the NASA C-MAPSS FD001
  turbofan dataset: feature engineering with verified causality, RUL
  regression (Linear/RandomForest/XGBoost), maintenance-risk classification
  tuned on F2, a FastAPI service with input validation and an OOD guard,
  and two frontends (live dashboard + static demo).
