"""Public API response schemas."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Readiness information."""

    status: Literal["healthy"]
    model: str
    threshold: float


class PredictionResponse(BaseModel):
    """Prediction returned for one chest radiograph."""

    request_id: str
    prediction: Literal["NORMAL", "PNEUMONIA"]
    pneumonia_probability: float = Field(ge=0.0, le=1.0)
    normal_probability: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    model: str
    gradcam_base64: str | None = None
    warning: str
