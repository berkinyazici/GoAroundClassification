from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INTERIM_DIR, PROCESSED_DIR
from src.features.build_features import TARGET_COLUMN, clean_data


def _profile_split(name: str, df: pd.DataFrame) -> None:
    rate = float(df[TARGET_COLUMN].mean()) if len(df) else 0.0
    print(f"{name}: {len(df):,} rows, go-around rate={rate:.6f}")
    if "time" in df.columns and len(df):
        print(f"  time: {df['time'].min()} -> {df['time'].max()}")


def _stratified_random_split(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stratify = df[TARGET_COLUMN] if df[TARGET_COLUMN].nunique() > 1 else None
    train, temp = train_test_split(df, test_size=0.30, random_state=seed, stratify=stratify)
    temp_stratify = temp[TARGET_COLUMN] if temp[TARGET_COLUMN].nunique() > 1 else None
    valid, test = train_test_split(temp, test_size=0.50, random_state=seed, stratify=temp_stratify)
    return train, valid, test


def make_splits(
    target: str = TARGET_COLUMN,
    sample_frac: float | None = None,
    seed: int = 42,
    split_mode: str = "time",
    source_path: Path | None = None,
) -> tuple[Path, Path, Path]:
    if target != TARGET_COLUMN:
        print(f"Using canonical target column '{TARGET_COLUMN}' instead of requested '{target}'.")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    source_path = source_path or INTERIM_DIR / "go_arounds_augmented.parquet"
    if not source_path.exists():
        legacy = INTERIM_DIR / "dataset.parquet"
        if legacy.exists():
            source_path = legacy
        else:
            raise FileNotFoundError(f"Interim dataset not found at {source_path}. Run make-interim first.")

    df = pd.read_parquet(source_path)
    df = clean_data(df)
    if sample_frac is not None:
        if not (0 < sample_frac <= 1):
            raise ValueError("--sample-frac must be in (0, 1].")
        df = df.sample(frac=sample_frac, random_state=seed).sort_index()

    if split_mode == "time" and "time" in df.columns and df["time"].notna().any():
        df = df.sort_values("time")
        train = df[df["time"] < pd.Timestamp("2019-09-01", tz="UTC")]
        valid = df[(df["time"] >= pd.Timestamp("2019-09-01", tz="UTC")) & (df["time"] < pd.Timestamp("2019-11-01", tz="UTC"))]
        test = df[df["time"] >= pd.Timestamp("2019-11-01", tz="UTC")]
        if min(len(train), len(valid), len(test)) == 0 or df[TARGET_COLUMN].nunique() < 2:
            print("Time split was not feasible for this dataset/sample; falling back to stratified random split.")
            train, valid, test = _stratified_random_split(df, seed)
    else:
        train, valid, test = _stratified_random_split(df, seed)

    paths = (PROCESSED_DIR / "train.parquet", PROCESSED_DIR / "valid.parquet", PROCESSED_DIR / "test.parquet")
    for split_df, path in zip((train, valid, test), paths):
        split_df.to_parquet(path, index=False)
    for name, split_df in zip(("train", "valid", "test"), (train, valid, test)):
        _profile_split(name, split_df)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/validation/test splits for go-around classification.")
    parser.add_argument("--target", default=TARGET_COLUMN)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["time", "random"], default="time")
    args = parser.parse_args()
    make_splits(target=args.target, sample_frac=args.sample_frac, seed=args.seed, split_mode=args.split_mode)


if __name__ == "__main__":
    main()
