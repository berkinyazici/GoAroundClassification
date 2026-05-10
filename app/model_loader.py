from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
from joblib import load
from sklearn.base import BaseEstimator

from src.config import MODEL_PATH


class DummyModel(BaseEstimator):
    def predict(self, X):
        return [0] * len(X)

    def predict_proba(self, X):
        return [[1.0, 0.0] for _ in range(len(X))]


class ModelService:
    def __init__(self, model_path: str | None = None) -> None:
        self.path = Path(model_path or MODEL_PATH)
        self.name = self.path.stem
        if self.path.exists():
            self.model = load(self.path)
        else:
            self.model = DummyModel()

    def predict(self, features: Dict[str, Any], model_name: str | None = None) -> Tuple[int, float]:
        if not isinstance(features, dict) or not features:
            raise ValueError("Features payload must be a non-empty dictionary.")

        df = pd.DataFrame([features])
        prediction = self.model.predict(df)[0]
        probability = 0.0
        if hasattr(self.model, "predict_proba"):
            probability = float(self.model.predict_proba(df)[0][-1])
        return int(prediction), probability
