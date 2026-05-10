from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INTERIM_DIR, PROCESSED_DIR


def make_splits(target: str = "target", test_size: float = 0.2, random_state: int = 42) -> tuple[Path, Path]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source_path = INTERIM_DIR / "dataset.csv"
    if not source_path.exists():
        raise FileNotFoundError(f"Interim dataset not found at {source_path}")

    print(f"Reading interim dataset from {source_path}")
    dataset = pd.read_csv(source_path, low_memory=False)
    print(f"Loaded interim dataset with shape {dataset.shape}")

    if target == 'has_ga':
        dataset[target] = dataset[target].replace(
            {True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1, 'false': 0}
        )
    if target not in dataset.columns:
        print(f"Available columns: {list(dataset.columns)}")
        raise ValueError(f"Target column '{target}' not found in interim data.")

    dataset[target] = pd.to_numeric(dataset[target], errors="coerce")
    missing_targets = dataset[target].isna().sum()
    if missing_targets > 0:
        print(f"Dropping {missing_targets} rows with missing '{target}' values before splitting.")
        dataset = dataset.dropna(subset=[target])

    y = dataset[target].astype(int)
    print(f"Using target column '{target}' with unique values: {sorted(y.unique().tolist())}")
    X = dataset.drop(columns=[target])
    stratify = y if len(y.unique()) > 1 else None

    print("Splitting data into train/test sets...")
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
    print(f"Train size: {len(X_train)} rows, Test size: {len(X_test)} rows")
    return train_path, test_path


if __name__ == "__main__":
    train_path, test_path = make_splits()
    print(f"Train split written to {train_path}")
    print(f"Test split written to {test_path}")
