from __future__ import annotations

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def random_split(df: pd.DataFrame, target_col: str = "go_around", seed: int = 42) -> SplitResult:
    train, temp = train_test_split(
        df,
        test_size=0.3,
        random_state=seed,
        stratify=df[target_col],
    )
    val, test = train_test_split(
        temp,
        test_size=0.5,
        random_state=seed,
        stratify=temp[target_col],
    )
    return SplitResult(train=train, val=val, test=test)


def group_split(
    df: pd.DataFrame,
    group_col: str,
    target_col: str = "go_around",
    seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> SplitResult:
    if group_col not in df.columns:
        raise ValueError(f"Group column '{group_col}' not in dataframe")

    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    idx_trainval, idx_test = next(gss1.split(df, groups=df[group_col]))
    trainval = df.iloc[idx_trainval]
    test = df.iloc[idx_test]

    remaining_val_ratio = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=remaining_val_ratio, random_state=seed)
    idx_train, idx_val = next(gss2.split(trainval, groups=trainval[group_col]))
    train = trainval.iloc[idx_train]
    val = trainval.iloc[idx_val]

    return SplitResult(train=train, val=val, test=test)


def time_split(
    df: pd.DataFrame,
    time_col: str,
    target_col: str = "go_around",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> SplitResult:
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not in dataframe")
    ordered = df.sort_values(time_col)
    n = len(ordered)
    i_train = int(n * train_ratio)
    i_val = int(n * (train_ratio + val_ratio))
    train = ordered.iloc[:i_train]
    val = ordered.iloc[i_train:i_val]
    test = ordered.iloc[i_val:]
    return SplitResult(train=train, val=val, test=test)
