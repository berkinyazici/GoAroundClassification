# Go-Around Classification

This repository is a complete Python project skeleton for binary go-around pattern recognition, including data preparation, model experimentation, FastAPI deployment, and a simple HTML frontend.

## Project structure

- `app/` — FastAPI backend, model loader, API schemas, and web UI assets.
- `src/` — experiment configuration, data engineering, feature building, model training, evaluation, and CLI.
- `data/` — local dataset storage for raw, interim, and processed files.
- `models/` — trained model artifacts.
- `reports/` — metrics, figures, and written documentation.
- `notebooks/` — exploratory notebooks.
- `tests/` — unit and integration test stubs.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Run data validation and preprocessing

```bash
python -m src.cli verify
python -m src.cli prepare
python -m src.cli split --target target
```

### Train a model

```bash
python -m src.cli train --model logreg --target target
```

### Evaluate the selected model

```bash
python -m src.cli evaluate --target target
```

### Serve the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker compose up --build
```

## API endpoints

- `GET /health` — health check.
- `GET /` — simple web interface.
- `POST /predict` — model prediction endpoint.

Example request:

```json
{
  "features": {
    "feature_1": 1.0,
    "feature_2": 0.5
  },
  "model_name": "logreg"
}
```

Example response:

```json
{
  "prediction": 0,
  "probability": 0.12,
  "model": "logreg"
}
```

## Notes

- The dataset is expected to exist locally under `data/raw/`.
- Training scripts write processed data under `data/processed/`.
- The default deployed model is loaded from `models/best_model.joblib`.
- Use `src/cli.py` to run verify, preprocessing, training, evaluation, and serve workflows.
