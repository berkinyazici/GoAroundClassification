"""Tests for model loading and prediction."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import FINAL_MODEL, FEATURE_SCHEMA


def test_final_model_exists():
    assert FINAL_MODEL.exists(), f"Missing {FINAL_MODEL}"


def test_feature_schema_exists():
    assert FEATURE_SCHEMA.exists(), f"Missing {FEATURE_SCHEMA}"


def test_model_loads():
    from src.models.common import load_model_bundle
    bundle = load_model_bundle(FINAL_MODEL)
    assert "model" in bundle
    assert "preprocessor" in bundle
    assert "feature_schema" in bundle


def test_model_predict_proba():
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline as _Pipeline
    from src.models.common import load_model_bundle

    bundle = load_model_bundle(FINAL_MODEL)
    model        = bundle["model"]
    preprocessor = bundle["preprocessor"]
    schema       = bundle["feature_schema"]

    num_feats = schema.get("numeric_features", [])
    cat_feats = schema.get("categorical_features", [])

    row = {f: np.nan for f in num_feats}
    row.update({f: "UNKNOWN" for f in cat_feats})
    X = pd.DataFrame([row])[num_feats + cat_feats]

    if isinstance(model, _Pipeline):
        proba = model.predict_proba(X)
    else:
        try:
            X_pre = preprocessor.transform(X)
        except Exception:
            X_pre = X.values
        proba = model.predict_proba(X_pre)

    assert proba.shape == (1, 2)
    assert abs(proba[0].sum() - 1.0) < 1e-5
    assert 0.0 <= proba[0, 1] <= 1.0


def test_model_loader_predict():
    from app.model_loader import predict
    result = predict({"airport": "EDDF", "wtc": "M", "wind_speed_knts": 12.0})
    assert "predicted_class" in result
    assert "probability_go_around" in result
    assert result["predicted_class"] in {0, 1}
    assert 0.0 <= result["probability_go_around"] <= 1.0
