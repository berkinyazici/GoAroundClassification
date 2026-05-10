from typing import Iterable

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

EXCLUDED_FEATURE_COLUMNS = {
    "ga_rate",
    "n_landings",
    "n_approaches",
    "n_rwy_approached",
    "time",
    "icao24",
    "callsign",
    "registration",
    "Folder:",
    "S:\\pools\\t\\T-ZAV-OSN-Data\\landings\\ga_verification3",
}

EXCLUDED_FEATURE_PREFIXES = ("Unnamed:",)


def resolve_target_column(df: pd.DataFrame, target_column: str = "target") -> str:
    if target_column in df.columns:
        return target_column
    if target_column == "target" and "has_ga" in df.columns:
        return "has_ga"
    raise ValueError(f"Target column '{target_column}' is missing from the dataset.")


def filter_feature_columns(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column not in EXCLUDED_FEATURE_COLUMNS
        and not any(column.startswith(prefix) for prefix in EXCLUDED_FEATURE_PREFIXES)
    ]


def build_features(df: pd.DataFrame, target_column: str = "target") -> tuple[pd.DataFrame, pd.Series, Pipeline]:
    target_column = resolve_target_column(df, target_column)

    feature_df = df.drop(columns=[target_column])
    feature_columns = filter_feature_columns(feature_df.columns)
    feature_df = feature_df[feature_columns]

    numeric_columns = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = feature_df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )

    pipeline = Pipeline([("transform", transformer)])
    X = pipeline.fit_transform(feature_df)
    y = df[target_column]
    return X, y, pipeline


def transform_features(df: pd.DataFrame, pipeline: Pipeline) -> pd.DataFrame:
    transformed = pipeline.transform(df)
    return pd.DataFrame(transformed)
