"""Response model for username lookup."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.predict import ProfileFeatures


class LookupData(BaseModel):
    username: str
    label: Literal["Fake", "Real"]
    confidence: float = Field(..., ge=0, le=1)
    probabilities: dict[str, float]
    profile: ProfileFeatures = Field(
        ...,
        description="The features the verdict was computed from. Returned so the "
                    "answer can be audited rather than trusted blindly.",
    )
    source: str = Field(
        ...,
        description="Which provider supplied the profile data. `mock` means the "
                    "numbers are synthetic and the verdict is meaningless.",
        examples=["mock", "http", "cached"],
    )
    model_version: str
    latency_ms: float
