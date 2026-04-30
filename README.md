# GoAroundClassification

Starter implementation for go-around risk classification using ADS-B + METAR style tabular features.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m goaround.train --data /path/to/go_arounds_augmented.csv.gz --target go_around --model logreg
uvicorn goaround.api.app:app --reload
```

## Project structure

- `src/goaround/data`: loading, cleaning, splitting
- `src/goaround/features`: preprocessing pipeline
- `src/goaround/models`: model factory (logreg, LDA, RF, MLP)
- `src/goaround/eval`: classification metrics
- `src/goaround/api`: FastAPI inference API
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
