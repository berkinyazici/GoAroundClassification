"""FastAPI application: serves the go-around classifier and HTML interface."""
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import PredictRequest, PredictResponse
from app.model_loader import calculate_derived_features, calculate_sensitivity, predict

BASE_DIR    = Path(__file__).resolve().parent
TEMPLATES   = Jinja2Templates(directory=str(BASE_DIR / "templates"))
PROJECT_ROOT = BASE_DIR.parent
FEATURE_IMPORTANCE_PATH = PROJECT_ROOT / "reports" / "metrics" / "final_feature_importance.json"

app = FastAPI(
    title="Go-Around Classifier",
    description="Binary classification of go-around risk using ADS-B and METAR features.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return TEMPLATES.TemplateResponse(request=request, name="index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "go-around-classifier"}


@app.post("/predict", response_model=PredictResponse)
async def predict_endpoint(body: PredictRequest):
    try:
        result = predict(body.model_dump())
        return PredictResponse(**result)
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/derived-features")
async def derived_features_endpoint(body: PredictRequest):
    try:
        return calculate_derived_features(body.model_dump())
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/feature-importance")
async def feature_importance_endpoint():
    if not FEATURE_IMPORTANCE_PATH.exists():
        return JSONResponse(
            status_code=404,
            content={"error": "Feature importance file not found. Run permutation importance first."},
        )
    with open(FEATURE_IMPORTANCE_PATH) as fh:
        return json.load(fh)


@app.post("/sensitivity")
async def sensitivity_endpoint(body: PredictRequest):
    try:
        return calculate_sensitivity(body.model_dump())
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
