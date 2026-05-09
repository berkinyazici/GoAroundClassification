from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_features(df: pd.DataFrame, target_column: str = "target") -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' is missing from the dataset.")

    numeric_columns = df.select_dtypes(include=["number"]).columns.drop(target_column).tolist()
    categorical_columns = df.select_dtypes(include=["object", "category"]).columns.tolist()

    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([("transform", transformer)])
    X = pipeline.fit_transform(df.drop(columns=[target_column]))
    y = df[target_column]
    return X, y, pipeline


def transform_features(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    transformed = pipeline.transform(df)
    return pd.DataFrame(transformed)
