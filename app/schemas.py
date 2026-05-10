from typing import Dict, Optional
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    features: Dict[str, float]
    model_name: Optional[str] = "logreg"


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model: str
