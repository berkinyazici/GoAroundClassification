from pathlib import Path

from joblib import dump
from src.config import MODEL_DIR, MODEL_PATH, PROCESSED_DIR
from src.models.common import get_classifier
from src.data.make_splits import make_splits
import pandas as pd


def train_logreg(target: str = "target") -> Path:
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Train split not found. Run src.data.make_splits first.")

    df = pd.read_parquet(train_path)
    X = df.drop(columns=[target])
    y = df[target]
    model = get_classifier("logreg")
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_PATH.with_name("logreg.joblib")
    dump(model, output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train logistic regression model")
    parser.add_argument("--target", default="target", help="Target column name")
    args = parser.parse_args()

    path = train_logreg(target=args.target)
    print(f"Saved logistic regression model to {path}")
