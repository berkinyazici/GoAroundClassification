"""Convert raw CSV.GZ files to Parquet for fast reloading."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import (
    AUGMENTED_CSV_GZ, AGG_CSV_GZ,
    AUGMENTED_PARQUET, AGG_PARQUET,
    DATA_INTERIM,
)


def _bool_col(series: pl.Series) -> pl.Series:
    """Coerce True/False/1/0/'true'/'false' strings to Int8."""
    if series.dtype == pl.Boolean:
        return series.cast(pl.Int8)
    try:
        return (
            series.cast(pl.Utf8)
            .str.to_lowercase()
            .map_elements(
                lambda v: 1 if v in {"true", "1", "yes"} else (0 if v in {"false", "0", "no"} else None),
                return_dtype=pl.Int8,
            )
        )
    except Exception:
        return series


def convert_augmented() -> None:
    print("Reading go_arounds_augmented.csv.gz ...")
    df = pl.read_csv(AUGMENTED_CSV_GZ, infer_schema_length=10000, ignore_errors=True)
    print(f"  Raw shape: {df.shape}")

    # Create binary target
    if "has_ga" in df.columns:
        df = df.with_columns(
            pl.col("has_ga").cast(pl.Utf8).str.to_lowercase()
            .map_elements(lambda v: 1 if v in {"true", "1"} else 0, return_dtype=pl.Int8)
            .alias("target")
        )

    # Parse time column
    if "time" in df.columns:
        try:
            df = df.with_columns(pl.col("time").str.to_datetime(strict=False))
        except Exception:
            pass

    # Normalize boolean column
    if "has_intersection" in df.columns:
        df = df.with_columns(_bool_col(df["has_intersection"]).alias("has_intersection"))

    DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    df.write_parquet(AUGMENTED_PARQUET)
    print(f"  Saved {AUGMENTED_PARQUET.name}: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # Target distribution
    if "target" in df.columns:
        pos = df["target"].sum()
        print(f"  Target: {pos:,} go-arounds ({100*pos/len(df):.3f}%)")

    # Missing value summary
    missing = {c: df[c].null_count() for c in df.columns if df[c].null_count() > 0}
    top20 = sorted(missing.items(), key=lambda x: -x[1])[:20]
    if top20:
        print("  Top missing columns:")
        for col, cnt in top20:
            print(f"    {col}: {cnt:,} ({100*cnt/len(df):.1f}%)")


def convert_agg() -> None:
    print("Reading go_arounds_agg.csv.gz ...")
    df = pl.read_csv(AGG_CSV_GZ, infer_schema_length=1000, ignore_errors=True)
    print(f"  Raw shape: {df.shape}")
    df.write_parquet(AGG_PARQUET)
    print(f"  Saved {AGG_PARQUET.name}")


def main() -> None:
    from src.data.verify_local_data import verify
    verify()
    convert_augmented()
    convert_agg()
    print("Interim conversion complete.")


if __name__ == "__main__":
    main()
