# Go-Around Classification Using ADS-B and METAR Data

This repository contains a submission-ready term project for **binary go-around classification**. It uses the public augmented landing dataset from Monstein et al. / Zenodo record 7148117, which combines ADS-B-derived landing information with aircraft, airport/runway, operator, and METAR weather attributes.

The project satisfies the course guideline requirements:

- a Pattern Recognition problem with a formal binary-classification target;
- reproducible preprocessing, feature engineering, splitting, model training, ablation, and evaluation scripts;
- implemented classifiers: Logistic Regression, LDA, Decision Tree / Random Forest-style tree baseline, MLP, and LightGBM;
- metrics for imbalanced classification: accuracy, precision, recall, F1, ROC-AUC, PR-AUC, and confusion matrix;
- a FastAPI backend serving a trained model artifact;
- a simple HTML client in the same Docker container;
- Dockerized demo workflow.

## Repository structure

```text
app/                    FastAPI backend and HTML client
configs/                ADS-B-only and ADS-B+METAR feature-set configs
scripts/run_ablation.py ADS-B-only vs ADS-B+METAR experiment runner
src/config.py           Project paths
src/data/               Download, verification, raw-to-parquet conversion, splitting
src/features/           Cleaning and feature schema
src/models/             Classifier factories, training wrappers, model selection
src/evaluation/         Final evaluation and figures
tests/                  Unit/API tests
reports/                Final report and generated metrics/figures (metrics/figures are gitignored)
```

## Dataset

Download these files from Zenodo record 7148117 into `data/raw/`:

- `go_arounds_augmented.csv.gz`
- `go_arounds_agg.csv.gz`
- `validation_table.xlsx`

You can download them with:

```bash
python -m src.cli download-data
```

The raw and processed data folders are intentionally gitignored because the dataset is large.

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## End-to-end experiment workflow

```bash
python -m src.cli verify
python -m src.cli make-interim
python -m src.cli make-splits --sample-frac 0.05
python -m src.cli train --model logreg --feature-set adsb_only
python -m src.cli train --model logreg --feature-set adsb_plus_metar
python -m src.cli train --model tree --feature-set adsb_plus_metar
python -m src.cli select-best
python -m src.cli evaluate
```

For the final full-data run, omit `--sample-frac`:

```bash
python -m src.cli make-splits
python scripts/run_ablation.py --models logreg tree lightgbm
python -m src.cli evaluate
```

Generated artifacts:

- trained model artifacts: `models/*.joblib`;
- deployed model artifact: `models/best_model.joblib`;
- metrics: `reports/metrics/*.json`;
- plots: `reports/figures/confusion_matrix.png`, `precision_recall_curve.png`, and `roc_curve.png`.

## API and web demo

Start locally:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>. The page accepts a JSON feature payload and displays prediction, label, probability, threshold, and model name.

API endpoints:

- `GET /health`
- `GET /`
- `POST /predict`

Example request:

```json
{
  "features": {
    "n_approaches": 1,
    "wind_speed_knts": 18,
    "wind_gust_knts": 28,
    "visibility_m": 3500,
    "airport": "EDDF",
    "runway": "25L",
    "typecode": "A320"
  }
}
```

## Docker demo

```bash
docker compose up --build
```

Then open <http://localhost:8000>. If `models/best_model.joblib` is mounted or included, the API serves the trained classifier. If not, the app uses a clearly named deterministic fallback model so that the container and UI remain demonstrable before the final training artifact is produced.

## Testing

```bash
pytest
```

## Project report

The final report draft is in `reports/final_report.md`. After running full experiments, paste the generated numerical values from `reports/metrics/evaluation_metrics.json` and exported figures from `reports/figures/` into the PDF submitted to the course system.
