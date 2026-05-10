"""Create time-based train/validation/test splits."""
import argparse
import sys
from pathlib import Path

import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import AUGMENTED_PARQUET, TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET, DATA_PROCESSED
from src.features.build_features import load_data, clean_data, add_time_features


def _print_split_stats(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    pos = int(df["target"].sum()) if "target" in df.columns else 0
    rate = 100.0 * pos / n if n else 0.0
    print(f"  {name}: {n:>10,} rows | {pos:>7,} go-arounds ({rate:.3f}%)")


def make_splits(sample_frac: float | None = None, top_airports: int | None = None) -> None:
    print(f"Loading {AUGMENTED_PARQUET.name} ...")
    raw = load_data(AUGMENTED_PARQUET)
    print(f"  Loaded {len(raw):,} rows, {raw.shape[1]} cols")

    if sample_frac is not None and sample_frac < 1.0:
        raw = raw.sample(frac=sample_frac, random_state=42)
        print(f"  Sampled {len(raw):,} rows (frac={sample_frac})")

    # Keep time column for splitting before cleaning drops it
    time_col = None
    if "time" in raw.columns:
        time_col = pd.to_datetime(raw["time"], errors="coerce")

    raw = add_time_features(raw)
    raw = clean_data(raw)

    # Re-attach time for splitting
    if time_col is not None:
        raw["_split_time"] = time_col.values[: len(raw)] if len(time_col) > len(raw) else time_col.reindex(raw.index).values

    if top_airports is not None and "airport" in raw.columns:
        top = (
            raw[raw["target"] == 1]["airport"]
            .value_counts()
            .head(top_airports)
            .index
        )
        raw = raw[raw["airport"].isin(top)]
        print(f"  Filtered to top {top_airports} airports: {len(raw):,} rows")

    if "_split_time" in raw.columns:
        t = pd.to_datetime(raw["_split_time"], errors="coerce")
        train = raw[t < "2019-09-01"].drop(columns=["_split_time"])
        valid = raw[(t >= "2019-09-01") & (t < "2019-11-01")].drop(columns=["_split_time"])
        test  = raw[t >= "2019-11-01"].drop(columns=["_split_time"])
    else:
        # Fallback: random 70/15/15 split if no time column
        print("  Warning: no time column found, using random 70/15/15 split")
        from sklearn.model_selection import train_test_split
        train, tmp = train_test_split(raw, test_size=0.30, random_state=42, stratify=raw["target"])
        valid, test = train_test_split(tmp,   test_size=0.50, random_state=42, stratify=tmp["target"])

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    pl.from_pandas(train.reset_index(drop=True)).write_parquet(TRAIN_PARQUET)
    pl.from_pandas(valid.reset_index(drop=True)).write_parquet(VALID_PARQUET)
    pl.from_pandas(test.reset_index(drop=True)).write_parquet(TEST_PARQUET)

    print("Split statistics:")
    for name, split in [("train", train), ("valid", valid), ("test", test)]:
        _print_split_stats(name, split)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train/valid/test splits")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--top-airports", type=int, default=None)
    args = parser.parse_args()
    make_splits(sample_frac=args.sample_frac, top_airports=args.top_airports)
    print("Splits saved to data/processed/.")


if __name__ == "__main__":
    main()
