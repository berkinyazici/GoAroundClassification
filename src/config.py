from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
BASE_DIR = ROOT_DIR.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
