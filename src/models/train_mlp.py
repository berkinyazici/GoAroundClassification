from pathlib import Path

from joblib import dump
from src.config import MODEL_DIR, MODEL_PATH, PROCESSED_DIR
from src.models.common import get_classifier
import pandas as pd


def train_mlp(target: str = "target") -> Path:
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Train split not found. Run src.data.make_splits first.")

    df = pd.read_parquet(train_path)
    X = df.drop(columns=[target])
    y = df[target]
    model = get_classifier("mlp")
    model.fit(X, y)
    output_path = MODEL_PATH.with_name("mlp.joblib")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MLP model")
    parser.add_argument("--target", default="target", help="Target column name")
    args = parser.parse_args()

    path = train_mlp(target=args.target)
    print(f"Saved MLP model to {path}")
