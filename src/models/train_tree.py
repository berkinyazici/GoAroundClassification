from pathlib import Path

from joblib import dump
from sklearn.metrics import accuracy_score
from src.config import MODEL_DIR, MODEL_PATH, PROCESSED_DIR
from src.models.common import get_classifier
import pandas as pd


def train_tree(target: str = "target") -> Path:
    train_path = PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise FileNotFoundError("Train split not found. Run src.data.make_splits first.")

    df = pd.read_parquet(train_path)
    X = df.drop(columns=[target])
    y = df[target]
    model = get_classifier("tree")
    print("Training Decision Tree...")
    model.fit(X, y)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    print(f"Train accuracy: {acc:.4f}")
    output_path = MODEL_PATH.with_name("tree.joblib")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dump(model, output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train decision tree model")
    parser.add_argument("--target", default="target", help="Target column name")
    args = parser.parse_args()

    path = train_tree(target=args.target)
    print(f"Saved decision tree model to {path}")
