"""Central path configuration for the project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_RAW        = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM    = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED  = PROJECT_ROOT / "data" / "processed"
MODELS_DIR      = PROJECT_ROOT / "models"
REPORTS_DIR     = PROJECT_ROOT / "reports"
FIGURES_DIR     = REPORTS_DIR / "figures"
METRICS_DIR     = REPORTS_DIR / "metrics"
ERROR_ANALYSIS_DIR = METRICS_DIR / "error_analysis"

# Dataset filenames
AUGMENTED_CSV_GZ   = DATA_RAW / "go_arounds_augmented.csv.gz"
AGG_CSV_GZ         = DATA_RAW / "go_arounds_agg.csv.gz"
VALIDATION_XLSX    = DATA_RAW / "validation_table.xlsx"

AUGMENTED_PARQUET  = DATA_INTERIM / "go_arounds_augmented.parquet"
AGG_PARQUET        = DATA_INTERIM / "go_arounds_agg.parquet"

TRAIN_PARQUET      = DATA_PROCESSED / "train.parquet"
VALID_PARQUET      = DATA_PROCESSED / "valid.parquet"
TEST_PARQUET       = DATA_PROCESSED / "test.parquet"

FINAL_MODEL        = MODELS_DIR / "final_model.joblib"
FEATURE_SCHEMA     = MODELS_DIR / "feature_schema.json"

# Ensure directories exist when imported
for _d in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, MODELS_DIR,
           FIGURES_DIR, METRICS_DIR, ERROR_ANALYSIS_DIR]:
    _d.mkdir(parents=True, exist_ok=True)
