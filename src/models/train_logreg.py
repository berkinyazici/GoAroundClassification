from pathlib import Path

from joblib import dump
from sklearn.metrics import accuracy_score
from src.config import MODEL_DIR, MODEL_PATH, PROCESSED_DIR
from src.models.common import get_classifier
import pandas as pd


def train_logreg(target: str = "target") -> Path:
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Train split not found. Run src.data.make_splits first.")

    df = pd.read_parquet(train_path)
    X = df.drop(columns=[target])
    y = df[target]
    model = get_classifier("logreg")
    print("Training Logistic Regression...")
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Train accuracy: {acc:.4f}")
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
