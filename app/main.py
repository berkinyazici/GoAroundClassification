"""FastAPI application: serves the go-around classifier and HTML interface."""
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.schemas import PredictRequest, PredictResponse
from app.model_loader import predict

BASE_DIR    = Path(__file__).resolve().parent
TEMPLATES   = Jinja2Templates(directory=str(BASE_DIR / "templates"))

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
