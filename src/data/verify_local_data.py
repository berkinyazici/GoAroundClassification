"""Verify local dataset files are present before preprocessing.

Only go_arounds_augmented.csv.gz is strictly required.
go_arounds_agg.csv.gz and validation_table.xlsx are generated automatically
from the augmented file if they are missing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.config import AUGMENTED_CSV_GZ, AGG_CSV_GZ, VALIDATION_XLSX, DATA_RAW


def _generate_optional_files() -> None:
    """Generate agg and validation files from the augmented CSV if missing."""
    if AGG_CSV_GZ.exists() and VALIDATION_XLSX.exists():
        return

    print("  Generating optional summary files from augmented dataset ...")
    import polars as pl
    import pandas as pd

    df = pl.read_csv(AUGMENTED_CSV_GZ, infer_schema_length=10000, ignore_errors=True)

    if not AGG_CSV_GZ.exists():
        has_ga_col = "has_ga" if "has_ga" in df.columns else None
        if has_ga_col and "airport" in df.columns and "runway" in df.columns:
            agg = (
                df.group_by(["airport", "runway"])
                .agg([
                    pl.len().alias("n_landings"),
                    pl.col(has_ga_col).cast(pl.Utf8).str.to_lowercase()
                    .map_elements(lambda v: 1 if v in {"true", "1"} else 0, return_dtype=pl.Int32)
                    .sum().alias("n_ga"),
                ])
            )
        else:
            agg = df.select(["airport", "runway"]).unique() if "airport" in df.columns else pl.DataFrame()
        agg.to_pandas().to_csv(AGG_CSV_GZ, index=False, compression="gzip")
        print(f"  Generated {AGG_CSV_GZ.name}")

    if not VALIDATION_XLSX.exists():
        try:
            pdf = df.select([c for c in ["airport", "has_ga"] if c in df.columns]).to_pandas()
            if "airport" in pdf.columns and "has_ga" in pdf.columns:
                summary = (
                    pdf.groupby("airport")["has_ga"]
                    .agg(["count", "sum"])
                    .reset_index()
                    .rename(columns={"count": "n_landings", "sum": "n_ga"})
                )
            else:
                summary = pd.DataFrame({"note": ["generated"]})
            summary.to_excel(VALIDATION_XLSX, index=False)
            print(f"  Generated {VALIDATION_XLSX.name}")
        except Exception as e:
            print(f"  Could not generate validation_table.xlsx: {e} (not required)")


def verify() -> None:
    if not AUGMENTED_CSV_GZ.exists():
        raise FileNotFoundError(
            f"Required file missing: {AUGMENTED_CSV_GZ}\n"
            "Place go_arounds_augmented.csv.gz under data/raw/ "
            "or run: python src/data/download_data.py"
        )

    print(f"  Found {AUGMENTED_CSV_GZ.name}  ({AUGMENTED_CSV_GZ.stat().st_size / 1e6:.1f} MB)")

    _generate_optional_files()

    print("Local dataset verification successful.")


def main() -> None:
    verify()


if __name__ == "__main__":
    main()
