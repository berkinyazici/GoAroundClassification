from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import joblib
import json
import pandas as pd
from pydantic import BaseModel
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import MODEL_DIR, PROCESSED_DIR, REPORT_DIR
from src.features.build_features import resolve_target_column

app = FastAPI(title="Go-Around Classification API")

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "lda": "Linear Discriminant Analysis",
    "tree": "Decision Tree",
    "mlp": "Multi-layer Perceptron",
    "lightgbm": "LightGBM",
}


def load_all_models() -> dict[str, Any]:
    models: dict[str, Any] = {}
    for model_file in MODEL_DIR.glob("*.joblib"):
        model_name = model_file.stem
        if model_name == "best_model":
            continue
        try:
            model = joblib.load(model_file)
            models[model_name] = model
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
    return models


def calculate_metrics(model: Any, target: str = "target") -> dict[str, float]:
    test_path = PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        return {}

    test_df = pd.read_parquet(test_path)
    target = resolve_target_column(test_df, target)
    y_true = test_df[target]
    X_test = test_df.drop(columns=[target])
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def load_all_metrics(models: dict[str, Any]) -> dict[str, dict[str, float]]:
    metrics_file = REPORT_DIR / "metrics" / "all_models_metrics.json"
    metrics: dict[str, dict[str, float]] = {}
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)

    missing_models = [model_name for model_name in models if model_name not in metrics]
    if not missing_models:
        return metrics

    for model_name in missing_models:
        try:
            metrics[model_name] = calculate_metrics(models[model_name])
        except Exception as e:
            print(f"Error calculating metrics for {model_name}: {e}")

    if metrics:
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def get_expected_features(model: Any) -> list[str]:
    feature_pipeline = getattr(model, "named_steps", {}).get("features")
    transformer = getattr(feature_pipeline, "named_steps", {}).get("transform")
    if transformer is None:
        return []
    return list(getattr(transformer, "feature_names_in_", []))


def build_prediction_frame(model: Any, payload: dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    expected_features = get_expected_features(model)
    for column in expected_features:
        if column not in df.columns:
            df[column] = float("nan")
    return df[expected_features] if expected_features else df


MODELS = load_all_models()
METRICS = load_all_metrics(MODELS)


class PredictionRequest(BaseModel):
    model_name: str
    data: dict[str, Any] | None = None
    features: dict[str, Any] | None = None


@app.get("/health")
def health_check():
    return {"status": "ok", "models_loaded": len(MODELS)}


@app.get("/")
def home():
    with open(Path(__file__).resolve().parent / "templates" / "index.html") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/models")
def get_models():
    """Tüm modelleri ve metriklerini döndür"""
    return {
        "models": [
            {"name": model_name, "label": MODEL_LABELS.get(model_name, model_name)}
            for model_name in MODELS
        ],
        "metrics": METRICS,
    }


@app.get("/api/metrics/{model_name}")
def get_model_metrics(model_name: str):
    """Spesifik model metriklerini döndür"""
    if model_name not in METRICS:
        raise HTTPException(status_code=404, detail=f"Metrics for {model_name} not found")
    return {"model": model_name, "metrics": METRICS[model_name]}


@app.post("/api/predict")
def predict(request: PredictionRequest):
    """Seçilen model ile tahmin yap"""
    model_name = request.model_name
    payload = request.data or request.features

    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    if not payload:
        raise HTTPException(status_code=400, detail="Prediction payload cannot be empty")

    try:
        model = MODELS[model_name]
        metrics = METRICS.get(model_name, {})

        # Tahmin yap
        df = build_prediction_frame(model, payload)
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0] if hasattr(model, "predict_proba") else None

        response = {
            "model": model_name,
            "prediction": int(prediction),
            "metrics": metrics,
        }
        if probability is not None:
            response["probability"] = {
                "no_goaround": float(probability[0]),
                "goaround": float(probability[1]),
            }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
def predict_legacy(request: PredictionRequest):
    return predict(request)
