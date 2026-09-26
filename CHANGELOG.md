# Changelog

All notable changes to this project are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## Documentation

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
