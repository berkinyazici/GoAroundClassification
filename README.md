# GoAroundClassification

Starter implementation for go-around risk classification using ADS-B + METAR style tabular features.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

# single training run
python -m goaround.train --data /path/to/go_arounds_augmented.csv.gz --target go_around --model logreg

# config-driven run (airport-aware split)
python -m goaround.train --data /path/to/go_arounds_augmented.csv.gz --config configs/adsb_only.yaml

# ablation
python scripts/run_ablation.py --data /path/to/go_arounds_augmented.csv.gz

# serve API
uvicorn goaround.api.app:app --reload
```

## What is implemented now

- Config-driven experiment runs (`--config` with YAML)
- Split modes: `random`, `airport` (group-based), `time` (chronological)
- Model families: Logistic Regression, LDA, Random Forest, MLP
- Metrics: PR-AUC, ROC-AUC, F1, Precision, Recall, Balanced Accuracy
- Run artifacts saved to timestamped run folders under `artifacts/<run_id>/`
- Ablation helper script: ADS-B only vs ADS-B + METAR

## Project structure

- `src/goaround/data`: loading, cleaning, splitting
- `src/goaround/features`: preprocessing pipeline
- `src/goaround/models`: model factory (logreg, LDA, RF, MLP)
- `src/goaround/eval`: classification metrics
- `src/goaround/api`: FastAPI inference API
- `configs/`: experiment configs (ablation feature sets)
- `scripts/`: helper scripts (ablation runner)
- `web/`: minimal static page

## API

- `GET /health`
- `POST /predict` with payload:

```json
{
  "features": {
    "airport": "KJFK",
    "wind_speed": 15.0,
    "visibility": 6000
  }
}
```

Response:

```json
{
  "prediction": 0,
  "probability": 0.12
}
```
