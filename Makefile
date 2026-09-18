# Convenience wrappers around the verified run commands.
# Every target activates the venv and sets PYTHONPATH, which are the two
# things most easily forgotten when running the pipeline by hand.

VENV := venv
PY   := $(VENV)/bin/python
RUN  := PYTHONPATH=. $(PY)
HOST := 127.0.0.1
API_PORT  := 8000
WEB_PORT  := 8080

.PHONY: help setup train train-reg train-clf api dashboard predict demo clean

help:
	@echo "make setup      - create venv and install dependencies"
	@echo "make train      - train all models (regression + classification)"
	@echo "make api        - start the FastAPI server on :$(API_PORT)"
	@echo "make dashboard  - serve the live frontend on :$(WEB_PORT) (needs the API)"
	@echo "make predict    - send one real prediction request to a running API"
	@echo "make demo       - rebuild the static demo page in docs/ (no API needed)"
	@echo "make clean      - remove __pycache__ and trained artifacts"

setup:
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r requirements.txt
	@echo "macOS note: XGBoost also needs OpenMP -> brew install libomp"

train: train-reg train-clf

train-reg:
	$(RUN) src/models/train_regression.py

train-clf:
	$(RUN) src/models/train_classification.py

api:
	PYTHONPATH=. $(VENV)/bin/uvicorn src.api.main:app --host $(HOST) --port $(API_PORT)

dashboard:
	$(PY) -m http.server $(WEB_PORT) --directory frontend

predict:
	$(RUN) scripts/sample_request.py

# Regenerates the hosted demo from the trained models, then inlines the
# payload into a single self-contained page served by GitHub Pages.
demo:
	$(RUN) scripts/export_demo_data.py
	$(RUN) scripts/build_demo.py

clean:
	find . -name __pycache__ -not -path './$(VENV)/*' -exec rm -rf {} + 2>/dev/null || true
	rm -f models_saved/*.joblib
