"""Feature engineering for go-around classification."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# --------------------------------------------------------------------------- #
# Feature groups                                                               #
# --------------------------------------------------------------------------- #

NUMERIC_CONTEXT = [
    # n_approaches and n_rwy_approached are EXCLUDED: they encode the number of
    # approaches a flight ultimately made, which is post-hoc information that
    # leaks the target (a go-around adds an extra approach → n_approaches >= 2).
    # They are used only for filtering calibration flights in clean_data().
    "glide_slope_angle",
    "rwy_length",
    "month",
    "day_of_week",
    "hour_utc",
]

NUMERIC_METAR = [
    "wind_speed_knts",
    "wind_dir_deg",
    "wind_gust_knts",
    "visibility_m",
    "temperature_deg",
    "press_sea_level_p",
    "press_p",
]

NUMERIC_ENGINEERED = [
    "runway_heading_deg",
    "headwind_knts",
    "tailwind_knts",
    "crosswind_knts",
    "gust_spread_knts",
    "low_visibility_flag",
    "very_low_visibility_flag",
    "strong_wind_flag",
    "strong_crosswind_flag",
    "tailwind_flag",
    "adverse_weather_flag",
    "airport_risk_train",
    "runway_risk_train",
    "typecode_risk_train",
]

CATEGORICAL_CONTEXT = [
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

CATEGORICAL_METAR = [
    "weather_intensity",
    "weather_precipitation",
    "weather_desc",
    "weather_obscuration",
    "weather_other",
]

# Columns that must never be model features (leakage / IDs)
DROP_COLS = {"has_ga", "icao24", "callsign", "registration", "time"}


def get_feature_groups() -> dict:
    return {
        "numeric_context":     NUMERIC_CONTEXT,
        "numeric_metar":       NUMERIC_METAR,
        "numeric_engineered":  NUMERIC_ENGINEERED,
        "categorical_context": CATEGORICAL_CONTEXT,
        "categorical_metar":   CATEGORICAL_METAR,
    }


def load_data(path: str | Path) -> pd.DataFrame:
    df = pl.read_parquet(path).to_pandas()
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    if "time" in df.columns:
        t = pd.to_datetime(df["time"], errors="coerce")
        df = df.copy()
        df["month"]       = t.dt.month.astype("float32")
        df["day_of_week"] = t.dt.dayofweek.astype("float32")
        df["hour_utc"]    = t.dt.hour.astype("float32")
    return df


def runway_to_heading(runway: object) -> float:
    if pd.isna(runway):
        return np.nan
    digits = "".join(ch for ch in str(runway).strip() if ch.isdigit())
    if not digits:
        return np.nan
    heading = int(digits[:2]) * 10
    return 360.0 if heading == 0 else float(heading)


def numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "runway" in df.columns:
        df["runway_heading_deg"] = df["runway"].map(runway_to_heading).astype("float32")

    if {"wind_dir_deg", "wind_speed_knts", "runway_heading_deg"}.issubset(df.columns):
        wind_dir = pd.to_numeric(df["wind_dir_deg"], errors="coerce")
        wind_speed = pd.to_numeric(df["wind_speed_knts"], errors="coerce")
        runway_heading = pd.to_numeric(df["runway_heading_deg"], errors="coerce")
        angle_rad = np.deg2rad(wind_dir - runway_heading)
        signed_headwind = wind_speed * np.cos(angle_rad)
        df["headwind_knts"] = np.maximum(signed_headwind, 0).astype("float32")
        df["tailwind_knts"] = np.maximum(-signed_headwind, 0).astype("float32")
        df["crosswind_knts"] = np.abs(wind_speed * np.sin(angle_rad)).astype("float32")

    if {"wind_gust_knts", "wind_speed_knts"}.issubset(df.columns):
        gust = pd.to_numeric(df["wind_gust_knts"], errors="coerce")
        speed = pd.to_numeric(df["wind_speed_knts"], errors="coerce")
        df["gust_spread_knts"] = np.maximum(gust - speed, 0).astype("float32")

    visibility = numeric_series(df, "visibility_m")
    df["low_visibility_flag"] = (visibility < 5000).fillna(False).astype("int8")
    df["very_low_visibility_flag"] = (visibility < 1500).fillna(False).astype("int8")

    wind_speed = numeric_series(df, "wind_speed_knts")
    crosswind = numeric_series(df, "crosswind_knts")
    tailwind = numeric_series(df, "tailwind_knts")
    df["strong_wind_flag"] = (wind_speed >= 25).fillna(False).astype("int8")
    df["strong_crosswind_flag"] = (crosswind >= 15).fillna(False).astype("int8")
    df["tailwind_flag"] = (tailwind >= 5).fillna(False).astype("int8")

    weather_cols = [
        "weather_intensity",
        "weather_precipitation",
        "weather_obscuration",
        "weather_other",
    ]
    weather_signal = pd.Series(False, index=df.index)
    for col in weather_cols:
        if col in df.columns:
            values = df[col].fillna("UNKNOWN").astype(str).str.upper()
            weather_signal |= ~values.isin({"", "UNKNOWN", "NAN", "NONE"})
    df["adverse_weather_flag"] = (
        weather_signal
        | (df["strong_wind_flag"] == 1)
        | (df["strong_crosswind_flag"] == 1)
        | (df["very_low_visibility_flag"] == 1)
    ).astype("int8")

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Require target
    if "target" not in df.columns and "has_ga" in df.columns:
        df["target"] = (
            df["has_ga"].astype(str).str.lower()
            .map({"true": 1, "1": 1, "false": 0, "0": 0})
            .astype("Int8")
        )
    df = df[df["target"].notna()].copy()
    df["target"] = df["target"].astype(int)

    # Filter likely training/calibration flights
    if "n_approaches" in df.columns:
        df = df[pd.to_numeric(df["n_approaches"], errors="coerce").fillna(1) <= 2]

    # Fill missing categoricals
    cat_cols = CATEGORICAL_CONTEXT + CATEGORICAL_METAR
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].fillna("UNKNOWN").astype(str)

    # Numeric missings stay as NaN for sklearn imputers

    df = add_engineered_features(df)

    # Drop identifier columns
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    return df


def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: Literal[
        "context_only",
        "context_metar",
        "context_metar_engineered",
    ] = "context_metar",
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Return X, y, numeric_features, categorical_features."""
    df = add_engineered_features(df)
    groups = get_feature_groups()

    num_feats  = groups["numeric_context"].copy()
    cat_feats  = groups["categorical_context"].copy()

    if feature_set == "context_metar":
        num_feats += groups["numeric_metar"]
        cat_feats += groups["categorical_metar"]
    elif feature_set == "context_metar_engineered":
        num_feats += groups["numeric_metar"] + groups["numeric_engineered"]
        cat_feats += groups["categorical_metar"]

    # Keep only columns that actually exist
    risk_features = {"airport_risk_train", "runway_risk_train", "typecode_risk_train"}
    num_feats = [c for c in num_feats if c in df.columns or c in risk_features]
    cat_feats = [c for c in cat_feats if c in df.columns]
    for c in risk_features.intersection(num_feats):
        if c not in df.columns:
            df = df.copy()
            df[c] = pd.NA

    y = df["target"].astype(int)
    X = df[num_feats + cat_feats].copy()

    return X, y, num_feats, cat_feats
