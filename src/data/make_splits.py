from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INTERIM_DIR, PROCESSED_DIR


def make_splits(target: str = "target", test_size: float = 0.2, random_state: int = 42) -> tuple[Path, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source_path = INTERIM_DIR / "dataset.parquet"
    if not source_path.exists():
        raise FileNotFoundError(f"Interim dataset not found at {source_path}")

    dataset = pd.read_parquet(source_path)
    if target not in dataset.columns:
        raise ValueError(f"Target column '{target}' not found in interim data.")

    X = dataset.drop(columns=[target])
    y = dataset[target]
    stratify = y if len(y.unique()) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_path = PROCESSED_DIR / "train.parquet"
    test_path = PROCESSED_DIR / "test.parquet"
    pd.concat([X_train, y_train], axis=1).to_parquet(train_path)
    pd.concat([X_test, y_test], axis=1).to_parquet(test_path)
    return train_path, test_path


if __name__ == "__main__":
    train_path, test_path = make_splits()
    print(f"Train split written to {train_path}")
    print(f"Test split written to {test_path}")
