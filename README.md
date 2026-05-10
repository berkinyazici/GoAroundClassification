# Go-Around Classification Using ADS-B and METAR Data

This repository contains the implementation of a course project developed for **BBL514E Pattern Recognition**, a graduate-level course in the **M.Sc. in Computer Science** program at **Istanbul Technical University**.

The objective of this study is to classify whether a landing attempt will result in a **go-around** by using publicly available **ADS-B trajectory data** and **METAR weather observations**. The project is formulated as a supervised binary classification problem and includes data preprocessing, feature engineering, model training, evaluation, and a lightweight deployment pipeline.

**BBL514E Pattern Recognition — Term Project**
Furkan Güney (704241023) · Alper Berkin Yazıcı (704241020)

---

## Overview

Binary classification of landing go-around risk using publicly available ADS-B-derived tabular features and METAR weather observations.

| Label | Meaning |
|-------|---------|
| `1`   | Go-Around (missed approach) |
| `0`   | Normal Landing |

The system exposes the trained model through a **FastAPI** backend and an **HTML web interface**, both running inside a single **Docker container**.

---

## Quick Start (Docker — Recommended for Demo)

```bash
docker build -t goaround-classifier .
docker run -p 8000:8000 goaround-classifier
# Open http://localhost:8000
```

---

## Dataset

**Source:** [Zenodo Record 7148117](https://zenodo.org/records/7148117) — *Large Landing Trajectory Dataset for Go-Around Analysis*

Download the files into `data/raw/`:

```bash
python src/data/download_data.py
# or manually place:
#   data/raw/go_arounds_augmented.csv.gz   (~173 MB)
#   data/raw/go_arounds_agg.csv.gz
#   data/raw/validation_table.xlsx
```

---

## Local Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

---

## Training Pipeline

```bash
# 1. Verify data files
python -m src.data.verify_local_data

# 2. Convert CSV.GZ → Parquet
python -m src.data.make_interim

# 3. Create time-based splits (use --sample-frac 0.05 for quick test)
python -m src.data.make_splits --sample-frac 0.05

# 4. Train all classifiers
python -m src.models.train_lda
python -m src.models.train_logreg
python -m src.models.train_tree
python -m src.models.train_mlp
python -m src.models.train_lightgbm

# 5. Select best model
python -m src.models.select_best_model

# 6. Final evaluation
python -m src.evaluation.evaluate_models
python -m src.evaluation.error_analysis
```

### CLI alternative

```bash
goaround make-splits --sample-frac 0.05
goaround train-lda
goaround train-logreg
goaround train-tree
goaround train-mlp
goaround train-lightgbm
goaround select-best-model
goaround evaluate
goaround error-analysis
```

---

## API Usage

Start the development server:

```bash
uvicorn app.main:app --reload
# or: goaround serve
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | HTML web interface |
| GET | `/health` | Health check |
| POST | `/predict` | Predict go-around probability |

### Example `curl` request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "airport": "EDDF",
    "runway": "25L",
    "wtc": "M",
    "wind_speed_knts": 12.0,
    "visibility_m": 8000.0,
    "temperature_deg": 15.0,
    "rwy_length": 4000.0,
    "hour_utc": 14.0,
    "month": 5.0,
    "day_of_week": 2.0
  }'
```

Example response:

```json
{
  "predicted_class": 0,
  "predicted_label": "Normal Landing",
  "probability_go_around": 0.0231,
  "probability_normal_landing": 0.9769,
  "threshold": 0.42
}
```

---

## Models Compared

| Model | Feature Set | Notes |
|-------|------------|-------|
| LDA | context_only / context_metar | Classical PR baseline; Gaussian class-conditional assumption |
| Logistic Regression | context_only / context_metar | Probabilistic linear baseline |
| Random Forest | context_only / context_metar | Tree-based ensemble |
| MLP | context_only / context_metar | Neural baseline (sklearn) |
| LightGBM | context_only / context_metar | Gradient-boosted trees (strongest) |

**Feature sets:**
- `context_only` — airport, runway, aircraft, time features (no weather)
- `context_metar` — context + METAR weather features (ablation study)

---

## Evaluation Metrics

- Accuracy, Precision, Recall, F1-score
- ROC-AUC, PR-AUC (Average Precision)
- Confusion Matrix
- Calibration Curve

Figures saved to `reports/figures/`, metrics to `reports/metrics/`.

---

## Tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
GoAroundClassification/
├── app/                    # FastAPI backend + HTML interface
│   ├── main.py
│   ├── schemas.py
│   ├── model_loader.py
│   ├── templates/index.html
│   └── static/
├── src/
│   ├── config.py
│   ├── data/               # download, verify, interim, splits
│   ├── features/           # feature engineering
│   ├── models/             # LDA, LogReg, RF, MLP, LightGBM, selection
│   ├── evaluation/         # plots, error analysis
│   └── cli.py
├── models/
│   ├── final_model.joblib  # best trained model (committed)
│   └── feature_schema.json
├── reports/
│   ├── figures/            # ROC, PR, confusion matrix, etc.
│   └── metrics/            # JSON metric files, comparison CSV
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## Academic Disclaimer

This system is developed for academic pattern recognition research purposes only. It must not be used for operational aviation decision making.

**Dataset citation:** Monstein et al., "Large Landing Trajectory Dataset for Go-Around Analysis," *Engineering Proceedings*, 2022. doi: 10.3390/engproc2022028002
