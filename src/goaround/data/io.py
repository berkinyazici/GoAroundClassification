from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffixes = ''.join(path.suffixes).lower()
    if suffixes.endswith('.parquet'):
        return pd.read_parquet(path)
    if suffixes.endswith('.csv') or suffixes.endswith('.csv.gz'):
        return pd.read_csv(path, compression='infer', low_memory=False)
    raise ValueError(f'Unsupported dataset format: {path}')


def basic_clean(df: pd.DataFrame, target_col: str = 'go_around') -> pd.DataFrame:
    df = df.copy()
    if target_col not in df.columns and 'target' in df.columns:
        df[target_col] = df['target']
    if target_col not in df.columns and 'has_ga' in df.columns:
        df[target_col] = df['has_ga'].astype(int)
    if target_col not in df.columns:
        raise ValueError(f'Missing target column: {target_col}')
    df = df.dropna(subset=[target_col]).copy()
    df[target_col] = df[target_col].astype(int)
    return df
