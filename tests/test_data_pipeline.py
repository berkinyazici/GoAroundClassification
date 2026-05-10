"""Tests for the data pipeline."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import (
    AUGMENTED_PARQUET, AGG_PARQUET,
    TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET,
)


def test_interim_augmented_exists():
    assert AUGMENTED_PARQUET.exists(), f"Missing {AUGMENTED_PARQUET}"


def test_interim_agg_exists():
    assert AGG_PARQUET.exists(), f"Missing {AGG_PARQUET}"


def test_processed_splits_exist():
    for p in [TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET]:
        assert p.exists(), f"Missing {p}"


def test_target_column_binary():
    import polars as pl
    df = pl.read_parquet(TRAIN_PARQUET)
    assert "target" in df.columns, "Missing 'target' column in train split"
    unique_vals = set(df["target"].unique().to_list())
    assert unique_vals.issubset({0, 1}), f"target has non-binary values: {unique_vals}"


def test_splits_have_rows():
    import polars as pl
    for p in [TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET]:
        df = pl.read_parquet(p)
        assert len(df) > 0, f"{p.name} is empty"


def test_splits_contain_positive_class():
    import polars as pl
    for p in [TRAIN_PARQUET, VALID_PARQUET, TEST_PARQUET]:
        df = pl.read_parquet(p)
        assert df["target"].sum() > 0, f"No positive samples in {p.name}"


def test_no_data_files_tracked_by_git():
    """Ensure .gitignore prevents large data files from being tracked."""
    gitignore = Path(__file__).resolve().parents[1] / ".gitignore"
    assert gitignore.exists()
    content = gitignore.read_text()
    assert "data/raw/*" in content
    assert "data/interim/*" in content
    assert "data/processed/*" in content
