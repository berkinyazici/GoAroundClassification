from pathlib import Path
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .model_loader import ModelService
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="Go-Around Classification API")

app.mount("/static", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
model_service = ModelService()


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    try:
        prediction, probability = model_service.predict(payload.features, payload.model_name)
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model=payload.model_name or model_service.name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
