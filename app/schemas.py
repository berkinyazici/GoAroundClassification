from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: dict[str, Any] = Field(..., description="One landing record encoded as feature name/value pairs.")
    model_name: str | None = Field(default=None, description="Optional display name; the deployed artifact is used for prediction.")


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    probability: float
    threshold: float
    model: str
