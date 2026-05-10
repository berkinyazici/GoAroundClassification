from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator

from src.config import MODEL_PATH


class DemoFallbackModel(BaseEstimator):
    """Deterministic fallback so the API stays demonstrable before training artifacts exist."""

    def predict_proba(self, X):
        rows = X.to_dict(orient="records") if hasattr(X, "to_dict") else list(X)
        probs = []
        for row in rows:
            wind = float(row.get("wind_speed_knts") or 0)
            gust = float(row.get("wind_gust_knts") or wind)
            visibility = float(row.get("visibility_m") or 10000)
            approaches = float(row.get("n_approaches") or 1)
            score = -4.2 + 0.045 * wind + 0.035 * gust - 0.00012 * visibility + 0.55 * max(approaches - 1, 0)
            p = 1.0 / (1.0 + pow(2.718281828, -score))
            probs.append([1.0 - p, p])
        return probs

    def predict(self, X):
        return [int(p[1] >= 0.5) for p in self.predict_proba(X)]


class ModelService:
    def __init__(self, model_path: str | None = None) -> None:
        configured = model_path or os.getenv("MODEL_PATH") or str(MODEL_PATH)
        self.path = Path(configured)
        self.name = self.path.stem
        self.threshold = 0.5
        if self.path.exists():
            loaded = load(self.path)
            if isinstance(loaded, dict) and "pipeline" in loaded:
                self.model = loaded["pipeline"]
                self.threshold = float(loaded.get("threshold", 0.5))
                self.name = self.path.stem
            else:
                self.model = loaded
        else:
            self.model = DemoFallbackModel()
            self.name = "demo_fallback_untrained"

    @property
    def loaded_real_model(self) -> bool:
        return self.path.exists()

    def predict(self, features: dict[str, Any], model_name: str | None = None) -> tuple[int, float, float, str]:
        if not isinstance(features, dict) or not features:
            raise ValueError("Features payload must be a non-empty dictionary.")
        df = pd.DataFrame([features])
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(df)[0][-1])
            prediction = int(probability >= self.threshold)
        else:
            prediction = int(self.model.predict(df)[0])
            probability = float(prediction)
        label = "go-around" if prediction == 1 else "normal landing"
        return prediction, probability, self.threshold, label
