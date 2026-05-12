"""Load the trained model bundle at startup."""
from __future__ import annotations
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import add_engineered_features
from src.models.common import apply_risk_encoders

BASE_DIR     = Path(__file__).resolve().parent.parent
MODELS_DIR   = BASE_DIR / "models"
MODEL_PATH   = MODELS_DIR / "final_model.joblib"
SCHEMA_PATH  = MODELS_DIR / "feature_schema.json"

_bundle      = None
_schema      = None
_risk_schema = None

DERIVED_FEATURE_LABELS = {
    "runway_heading_deg": "Runway heading",
    "headwind_knts": "Headwind",
    "tailwind_knts": "Tailwind",
    "crosswind_knts": "Crosswind",
    "gust_spread_knts": "Gust spread",
    "low_visibility_flag": "Low visibility flag",
    "very_low_visibility_flag": "Very low visibility flag",
    "strong_wind_flag": "Strong wind flag",
    "strong_crosswind_flag": "Strong crosswind flag",
    "tailwind_flag": "Tailwind flag",
    "adverse_weather_flag": "Adverse weather flag",
    "airport_risk_train": "Airport historical risk",
    "runway_risk_train": "Runway historical risk",
    "typecode_risk_train": "Aircraft type historical risk",
}

DERIVED_FEATURE_DEPENDENCIES = {
    "runway_heading_deg": ["runway"],
    "headwind_knts": ["runway", "wind_dir_deg", "wind_speed_knts"],
    "tailwind_knts": ["runway", "wind_dir_deg", "wind_speed_knts"],
    "crosswind_knts": ["runway", "wind_dir_deg", "wind_speed_knts"],
    "gust_spread_knts": ["wind_gust_knts", "wind_speed_knts"],
    "low_visibility_flag": ["visibility_m"],
    "very_low_visibility_flag": ["visibility_m"],
    "strong_wind_flag": ["wind_speed_knts"],
    "strong_crosswind_flag": ["runway", "wind_dir_deg", "wind_speed_knts"],
    "tailwind_flag": ["runway", "wind_dir_deg", "wind_speed_knts"],
    "adverse_weather_flag": [
        "visibility_m",
        "wind_speed_knts",
        "wind_dir_deg",
        "runway",
        "weather_intensity",
        "weather_precipitation",
        "weather_obscuration",
        "weather_other",
    ],
    "airport_risk_train": ["airport"],
    "runway_risk_train": ["runway"],
    "typecode_risk_train": ["typecode"],
}

SENSITIVITY_NUMERIC_FIELDS = {
    "glide_slope_angle": 0.1,
    "rwy_length": 100.0,
    "month": 1.0,
    "day_of_week": 1.0,
    "hour_utc": 1.0,
    "wind_speed_knts": 5.0,
    "wind_dir_deg": 10.0,
    "wind_gust_knts": 5.0,
    "visibility_m": 1000.0,
    "temperature_deg": 5.0,
    "press_sea_level_p": 5.0,
    "press_p": 5.0,
}


def get_bundle():
    global _bundle, _schema
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
        if SCHEMA_PATH.exists():
            with open(SCHEMA_PATH) as fh:
                _schema = json.load(fh)
        else:
            _schema = _bundle.get("feature_schema", {})
    return _bundle, _schema


def get_risk_schema() -> dict:
    global _risk_schema
    if _risk_schema is not None:
        return _risk_schema

    _, schema = get_bundle()
    _risk_schema = schema.get("risk_schema", {})
    if _risk_schema:
        return _risk_schema

    for candidate in MODELS_DIR.glob("*context_metar_engineered.joblib"):
        try:
            bundle = joblib.load(candidate)
            _risk_schema = bundle.get("feature_schema", {}).get("risk_schema", {})
            if _risk_schema:
                return _risk_schema
        except Exception:
            continue

    _risk_schema = {}
    return _risk_schema


def calculate_derived_features(features: dict) -> dict:
    feature_df = pd.DataFrame([features])
    feature_df = add_engineered_features(feature_df)
    feature_df = apply_risk_encoders(feature_df, get_risk_schema())
    row = feature_df.iloc[0].to_dict()

    values = {}
    for name in DERIVED_FEATURE_LABELS:
        value = row.get(name)
        values[name] = None if value is None or pd.isna(value) else float(value)

    return {
        "values": values,
        "metadata": {
            name: {
                "label": DERIVED_FEATURE_LABELS[name],
                "depends_on": DERIVED_FEATURE_DEPENDENCIES[name],
            }
            for name in DERIVED_FEATURE_LABELS
        },
    }


def predict(features: dict) -> dict:
    bundle, schema = get_bundle()
    model        = bundle["model"]
    preprocessor = bundle["preprocessor"]

    num_feats = schema.get("numeric_features", [])
    cat_feats = schema.get("categorical_features", [])
    risk_schema = schema.get("risk_schema", {})
    threshold  = float(schema.get("best_threshold", 0.5))

    derived = calculate_derived_features(features)
    engineered_features = {**features, **derived["values"]}

    row = {}
    for f in num_feats:
        val = engineered_features.get(f)
        row[f] = float(val) if val is not None and not pd.isna(val) else np.nan
    for f in cat_feats:
        val = engineered_features.get(f)
        row[f] = str(val) if val is not None and not pd.isna(val) else "UNKNOWN"

    X = pd.DataFrame([row])[num_feats + cat_feats]

    from sklearn.pipeline import Pipeline as _Pipeline
    if isinstance(model, _Pipeline):
        proba = model.predict_proba(X)[0]
    else:
        try:
            X_pre = preprocessor.transform(X)
        except Exception:
            X_pre = X.values
        proba = model.predict_proba(X_pre)[0]
    prob_ga = float(proba[1])
    prob_nl = float(proba[0])
    pred_class = int(prob_ga >= threshold)
    label = "Go-Around Risk" if pred_class == 1 else "Normal Landing"

    return {
        "predicted_class":          pred_class,
        "predicted_label":          label,
        "probability_go_around":    round(prob_ga, 4),
        "probability_normal_landing": round(prob_nl, 4),
        "threshold":                threshold,
        "derived_features":          derived,
    }


def calculate_sensitivity(features: dict) -> dict:
    base_probability = predict(features)["probability_go_around"]
    records = []

    for field, default_step in SENSITIVITY_NUMERIC_FIELDS.items():
        raw_value = features.get(field)
        if raw_value is None:
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            continue

        step = max(abs(value) * 0.1, default_step)
        plus_features = dict(features)
        minus_features = dict(features)
        plus_features[field] = value + step
        minus_features[field] = value - step

        plus_probability = predict(plus_features)["probability_go_around"]
        minus_probability = predict(minus_features)["probability_go_around"]
        records.append({
            "feature": field,
            "step": round(step, 4),
            "base_probability": base_probability,
            "plus_probability": plus_probability,
            "plus_delta": round(plus_probability - base_probability, 4),
            "minus_probability": minus_probability,
            "minus_delta": round(minus_probability - base_probability, 4),
            "max_abs_delta": round(
                max(abs(plus_probability - base_probability), abs(minus_probability - base_probability)),
                4,
            ),
        })

    records.sort(key=lambda item: item["max_abs_delta"], reverse=True)
    return {"base_probability": base_probability, "items": records}
