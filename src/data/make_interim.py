from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import INTERIM_DIR, RAW_DIR
from src.data.verify_local_data import verify_local_data
from src.features.build_features import TARGET_COLUMN, clean_data, load_augmented_data


def _print_profile(df: pd.DataFrame, name: str) -> None:
    print(f"{name}: {len(df):,} rows x {len(df.columns):,} columns")
    if TARGET_COLUMN in df.columns:
        counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
        rate = float(df[TARGET_COLUMN].mean()) if len(df) else 0.0
        print(f"target distribution: {counts}; go-around rate={rate:.6f}")
    missing = df.isna().mean().sort_values(ascending=False).head(10)
    print("top missing fractions:")
    for col, frac in missing.items():
        print(f"  {col}: {frac:.3f}")


def make_interim(raw_dir: Path = RAW_DIR, interim_dir: Path = INTERIM_DIR) -> Path:
    interim_dir.mkdir(parents=True, exist_ok=True)
    verify_local_data(raw_dir, strict=True)

    augmented_raw = raw_dir / "go_arounds_augmented.csv.gz"
    augmented = clean_data(load_augmented_data(augmented_raw))
    augmented_path = interim_dir / "go_arounds_augmented.parquet"
    augmented.to_parquet(augmented_path, index=False)
    _print_profile(augmented, "augmented")

    agg_raw = raw_dir / "go_arounds_agg.csv.gz"
    if agg_raw.exists():
        agg = pd.read_csv(agg_raw, compression="infer", low_memory=False)
        agg_path = interim_dir / "go_arounds_agg.parquet"
        agg.to_parquet(agg_path, index=False)
        print(f"aggregate table: {len(agg):,} rows x {len(agg.columns):,} columns -> {agg_path}")

    return augmented_path


if __name__ == "__main__":
    output = make_interim()
    print(f"Interim augmented dataset written to {output}")
