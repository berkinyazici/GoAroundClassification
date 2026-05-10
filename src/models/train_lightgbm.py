from pathlib import Path

from joblib import dump
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from src.config import MODEL_DIR, MODEL_PATH, PROCESSED_DIR
from src.features.build_features import build_features
from src.models.common import get_classifier
import pandas as pd


def train_lightgbm(target: str = "target") -> Path:
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Train split not found. Run src.data.make_splits first.")

    df = pd.read_parquet(train_path)
    X, y, feature_pipeline = build_features(df, target)
    model = get_classifier("lightgbm")
    print("Training LightGBM...")
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Train accuracy: {acc:.4f}")
    full_pipeline = Pipeline([("features", feature_pipeline), ("model", model)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    output_path = MODEL_PATH.with_name("lightgbm.joblib")
    dump(full_pipeline, output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--target", default="target", help="Target column name")
    args = parser.parse_args()

    path = train_lightgbm(target=args.target)
    print(f"Saved LightGBM model to {path}")
