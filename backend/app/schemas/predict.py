"""Request/response models for the prediction endpoints.

Field names use snake_case on the wire (Dart/Flutter convention, and JSON keys
like `"Engagement Rate (%)"` from the original Flask form are painful in Dart).
`serialization_alias`/`validation_alias` keep the wire contract stable while the
service maps to the model's own column names in one place -- see
`app.services.model_service.FEATURE_COLUMNS`.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ProfileFeatures(BaseModel):
    """One social profile to classify.

    Bounds are enforced here rather than in a hand-rolled loop of lambdas the way
    `website/app.py:predict` did it, so validation errors are structured and the
    limits show up automatically in the OpenAPI schema.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "followers": 5000,
                "following": 300,
                "posts": 150,
                "engagement_rate": 4.5,
                "avg_likes_per_post": 400,
                "avg_comments_per_post": 20,
                "verified": False,
                "account_age_years": 5,
                "bio_text": "Foodie | Reviews and recipes",
                "full_name": "Priya Kumar",
                "profile_picture": True,
                "external_url": False,
                "language": "English",
            }
        }
    )

    followers: int = Field(..., ge=0, le=10_000_000_000, description="Follower count.")
    following: int = Field(..., ge=0, le=10_000_000_000, description="Accounts followed.")
    posts: int = Field(..., ge=0, le=10_000_000, description="Total posts published.")
    engagement_rate: float = Field(
        ..., ge=0, le=100, description="Engagement rate as a percentage (0-100)."
    )
    avg_likes_per_post: int = Field(..., ge=0, le=1_000_000_000)
    avg_comments_per_post: int = Field(..., ge=0, le=1_000_000_000)
    verified: bool = Field(..., description="Platform verification badge.")
    account_age_years: float = Field(..., ge=0, le=100)
    bio_text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Profile biography. Free text -- any string is accepted.",
    )

    # ---- profile attributes from the paper's Table 1 ----------------------
    # Optional so that a client written against the previous contract keeps
    # working. An omitted field is filled with the value the training data
    # imputed for it (see `ModelService._to_engineered_frame`), and every field
    # filled that way is named in `imputed_fields` on the response — a
    # prediction made from defaults should not look like one made from data.
    full_name: str | None = Field(
        None, max_length=200,
        description="Display name. Drives len_fullname, fullname_words and "
                    "ratio_numlen_fullname.",
    )
    profile_picture: bool | None = Field(
        None, description="Whether the account has a profile picture."
    )
    external_url: bool | None = Field(
        None, description="Whether the profile links out to an external URL."
    )
    language: str | None = Field(
        None, max_length=50, description="Profile language, if the platform reports one."
    )
    engagement_consistency: float | None = Field(
        None, ge=0, le=100,
        description="Variance-based consistency of engagement across posts, 0-100.",
    )

    @field_validator("bio_text")
    @classmethod
    def _strip_and_require(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("bio_text must not be blank")
        return v


class PredictionData(BaseModel):
    label: Literal["Fake", "Real"]
    confidence: float = Field(..., ge=0, le=1, description="Probability of the returned label.")
    probabilities: dict[str, float] = Field(
        ..., description="Probability per class, keyed by label."
    )
    model_version: str
    latency_ms: float
    imputed_fields: list[str] = Field(
        default_factory=list,
        description="Optional inputs that were not supplied and were filled with the "
                    "training-set default before scoring. A non-empty list means part "
                    "of this prediction rests on assumptions, not on the profile.",
    )


class BatchPredictRequest(BaseModel):
    """Batch scoring.

    Mobile clients that score a list of profiles should send one batch request
    instead of N single ones -- a batched `predict_proba` amortises the per-call
    numpy/pandas overhead, which dominates single-row inference.
    """

    items: list[ProfileFeatures] = Field(..., min_length=1, max_length=100)


class BatchPredictionItem(BaseModel):
    index: int
    label: Literal["Fake", "Real"]
    confidence: float
    probabilities: dict[str, float]
    imputed_fields: list[str] = Field(default_factory=list)


class BatchPredictionData(BaseModel):
    results: list[BatchPredictionItem]
    count: int
    model_version: str
    latency_ms: float
