from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TARGET_COLUMN = "target"
RAW_TARGET_COLUMN = "has_ga"

LEAKAGE_COLUMNS = {
    RAW_TARGET_COLUMN,
    "go_around",
    "is_go_around",
    "icao24",
    "callsign",
    "registration",
    "flight_id",
    "landing_id",
}

BASE_NUMERIC_FEATURES = [
    "n_approaches",
    "n_rwy_approached",
    "glide_slope_angle",
    "rwy_length",
    "wind_speed_knts",
    "wind_dir_deg",
    "wind_gust_knts",
    "visibility_m",
    "temperature_deg",
    "press_sea_level_p",
    "press_p",
    "month",
    "day_of_week",
    "hour_utc",
]

BASE_CATEGORICAL_FEATURES = [
    "airport",
    "runway",
    "typecode",
    "icaoaircrafttype",
    "wtc",
    "has_intersection",
    "airport_country",
    "airport_region",
    "operator_country",
    "operator_region",
    "weather_intensity",
    "weather_precipitation",
    "weather_desc",
    "weather_obscuration",
    "weather_other",
]

ADS_B_NUMERIC_FEATURES = [
    "n_approaches",
    "n_rwy_approached",
    "glide_slope_angle",
    "rwy_length",
    "month",
    "day_of_week",
    "hour_utc",
]

ADS_B_CATEGORICAL_FEATURES = [
    "airport",
    "runway",
    "typecode",
    "icaoaircrafttype",
    "wtc",
    "has_intersection",
    "airport_country",
    "airport_region",
    "operator_country",
    "operator_region",
]

METAR_NUMERIC_FEATURES = [
    "wind_speed_knts",
    "wind_dir_deg",
    "wind_gust_knts",
    "visibility_m",
    "temperature_deg",
    "press_sea_level_p",
    "press_p",
]

METAR_CATEGORICAL_FEATURES = [
    "weather_intensity",
    "weather_precipitation",
    "weather_desc",
    "weather_obscuration",
    "weather_other",
]


def _parse_bool_like(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, np.integer, float, np.floating)):
        return float(value > 0)
    text = str(value).strip().lower()
    if text in {"true", "t", "yes", "y", "1", "go-around", "go_around", "ga"}:
        return 1.0
    if text in {"false", "f", "no", "n", "0", "landing", "normal"}:
        return 0.0
    return np.nan


def _safe_weather_string(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "UNKNOWN"
    if isinstance(value, (list, tuple, set)):
        return "|".join(str(v).strip() for v in value if str(v).strip()) or "UNKNOWN"
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return "UNKNOWN"
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return "|".join(str(v).strip() for v in parsed if str(v).strip()) or "UNKNOWN"
        except (ValueError, SyntaxError):
            pass
    return text


def load_augmented_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        return pd.read_parquet(path)
    if suffixes.endswith(".csv") or suffixes.endswith(".csv.gz"):
        return pd.read_csv(path, compression="infer", low_memory=False)
    raise ValueError(f"Unsupported dataset format: {path}")


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        df["month"] = df["time"].dt.month
        df["day_of_week"] = df["time"].dt.dayofweek
        df["hour_utc"] = df["time"].dt.hour
    else:
        for col in ("month", "day_of_week", "hour_utc"):
            if col not in df.columns:
                df[col] = np.nan
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if TARGET_COLUMN not in df.columns:
        source_target = RAW_TARGET_COLUMN if RAW_TARGET_COLUMN in df.columns else None
        if source_target is None and "go_around" in df.columns:
            source_target = "go_around"
        if source_target is None:
            raise ValueError("Expected a target column named 'has_ga', 'go_around', or 'target'.")
        df[TARGET_COLUMN] = df[source_target].map(_parse_bool_like)
    else:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map(_parse_bool_like)

    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    if "n_approaches" in df.columns:
        df["n_approaches"] = pd.to_numeric(df["n_approaches"], errors="coerce")
        df = df[(df["n_approaches"].isna()) | (df["n_approaches"] <= 2)].copy()

    df = add_time_features(df)

    for col in BASE_NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in BASE_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].map(_safe_weather_string).astype("object")

    return df


def get_feature_columns(feature_set: str = "adsb_plus_metar", available_columns: list[str] | None = None) -> tuple[list[str], list[str]]:
    if feature_set == "adsb_only":
        numeric = ADS_B_NUMERIC_FEATURES.copy()
        categorical = ADS_B_CATEGORICAL_FEATURES.copy()
    elif feature_set == "metar_only":
        numeric = METAR_NUMERIC_FEATURES.copy()
        categorical = METAR_CATEGORICAL_FEATURES.copy()
    elif feature_set == "adsb_plus_metar":
        numeric = BASE_NUMERIC_FEATURES.copy()
        categorical = BASE_CATEGORICAL_FEATURES.copy()
    else:
        raise ValueError("feature_set must be one of: adsb_only, metar_only, adsb_plus_metar")

    if available_columns is not None:
        available = set(available_columns)
        numeric = [c for c in numeric if c in available]
        categorical = [c for c in categorical if c in available]
    return numeric, categorical


def split_X_y(df: pd.DataFrame, feature_set: str = "adsb_plus_metar") -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    numeric, categorical = get_feature_columns(feature_set, list(df.columns))
    features = numeric + categorical
    if not features:
        raise ValueError("No usable feature columns were found in the dataframe.")
    return df[features].copy(), df[TARGET_COLUMN].copy(), numeric, categorical
