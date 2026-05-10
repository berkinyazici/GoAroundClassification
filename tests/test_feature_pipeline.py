from __future__ import annotations

import pandas as pd

from src.features.build_features import clean_data, split_X_y
from src.models.common import build_model_pipeline


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": ["2019-01-01T12:00:00Z", "2019-10-01T13:00:00Z", "2019-12-01T14:00:00Z", "2019-12-02T15:00:00Z"],
            "has_ga": [0, 1, 0, 1],
            "n_approaches": [1, 1, 2, 1],
            "n_rwy_approached": [1, 1, 1, 2],
            "glide_slope_angle": [3.0, 3.1, 2.9, 3.2],
            "rwy_length": [3000, 3500, 2800, 3200],
            "wind_speed_knts": [5, 22, 8, 30],
            "wind_dir_deg": [180, 240, 200, 260],
            "wind_gust_knts": [7, 35, 10, 42],
            "visibility_m": [10000, 3000, 9000, 2000],
            "temperature_deg": [10, 12, 9, 8],
            "press_sea_level_p": [1015, 1008, 1012, 1005],
            "press_p": [1012, 1004, 1009, 1001],
            "airport": ["EDDF", "EDDF", "EHAM", "EHAM"],
            "runway": ["25L", "25L", "18R", "18R"],
            "typecode": ["A320", "A320", "B738", "B738"],
            "icaoaircrafttype": ["L2J", "L2J", "L2J", "L2J"],
            "wtc": ["M", "M", "M", "M"],
            "has_intersection": [False, False, True, True],
            "airport_country": ["DE", "DE", "NL", "NL"],
            "airport_region": ["Europe", "Europe", "Europe", "Europe"],
            "operator_country": ["DE", "DE", "NL", "NL"],
            "operator_region": ["Europe", "Europe", "Europe", "Europe"],
            "weather_intensity": [None, "+", None, "-"],
            "weather_precipitation": ["UNKNOWN", "RA", "UNKNOWN", "RA"],
            "weather_desc": ["UNKNOWN", "SH", "UNKNOWN", "TS"],
            "weather_obscuration": ["UNKNOWN", "BR", "UNKNOWN", "BR"],
            "weather_other": ["UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN"],
        }
    )


def test_clean_data_adds_target_and_time_features():
    df = clean_data(sample_frame())
    assert set(df["target"].unique()) == {0, 1}
    assert {"month", "day_of_week", "hour_utc"}.issubset(df.columns)
    assert str(df["time"].dtype).startswith("datetime64")


def test_pipeline_fits_sample_data():
    df = clean_data(sample_frame())
    X, y, numeric, categorical = split_X_y(df)
    model = build_model_pipeline("logreg", numeric, categorical)
    model.fit(X, y)
    probabilities = model.predict_proba(X)[:, 1]
    assert len(probabilities) == len(df)
