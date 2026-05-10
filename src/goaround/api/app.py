from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: Dict[str, Any]


def create_app(model_path: str = "artifacts/model.joblib") -> FastAPI:
    app = FastAPI(title="Go-Around Risk API")

    mp = Path(model_path)
    model = joblib.load(mp) if mp.exists() else None

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": model is not None}

    @app.post("/predict")
    def predict(req: PredictRequest):
        if model is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        row = [req.features]
        proba = float(model.predict_proba(row)[0, 1])
        pred = int(proba >= 0.5)
        return {"prediction": pred, "probability": proba}

    return app


app = create_app()
