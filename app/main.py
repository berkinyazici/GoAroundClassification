from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .model_loader import ModelService
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Go-Around Classification API", version="1.0.0")

APP_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
model_service = ModelService()


@app.get("/health")
def health_check():
    return {"status": "ok", "model": model_service.name, "real_model_loaded": model_service.loaded_real_model}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "model_name": model_service.name})


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        prediction, probability, threshold, label = model_service.predict(payload.features, payload.model_name)
        return PredictionResponse(
            prediction=prediction,
            label=label,
            probability=probability,
            threshold=threshold,
            model=model_service.name,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
