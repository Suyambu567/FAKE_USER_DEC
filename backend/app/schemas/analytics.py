"""Analytics + model-metadata response models.

Everything here is computed once at startup (see `AnalyticsService`), not on every
request the way `website/app.py:analytics` re-read a 15k-row CSV per page load.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FeatureImportance(BaseModel):
    feature: str
    importance: float


class ClassDistribution(BaseModel):
    label: str
    count: int
    percentage: float


class HistogramBin(BaseModel):
    label: str
    count: int


class ModelMetrics(BaseModel):
    """Honest metrics, straight from the training run.

    `baseline_accuracy` is the majority-class rate. If `accuracy` is not clearly
    above it, the model has no predictive power -- which is exactly the case for
    the dataset shipped with this project. Exposing both makes that impossible to
    hide behind a single impressive-looking number.
    """

    accuracy: float
    baseline_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    lift_over_baseline: float = Field(
        ..., description="accuracy - baseline_accuracy. <= 0 means the model is useless."
    )


class AnalyticsData(BaseModel):
    training_samples: int
    feature_count: int
    class_distribution: list[ClassDistribution]
    feature_importances: list[FeatureImportance]
    engagement_histogram: list[HistogramBin]
    metrics: ModelMetrics
    model_version: str
    trained_at: str


class ModelInfoData(BaseModel):
    model_version: str
    algorithm: str
    trained_at: str
    sklearn_version: str
    feature_count: int
    classes: list[str]
    metrics: ModelMetrics
    warnings: list[str] = Field(
        default_factory=list,
        description="Data-quality or model-quality caveats detected at training time.",
    )


class FeatureSpec(BaseModel):
    name: str
    type: str
    required: bool
    minimum: float | None = None
    maximum: float | None = None
    description: str


class FeatureSchemaData(BaseModel):
    """Lets a Flutter client build its form dynamically instead of hardcoding fields."""

    features: list[FeatureSpec]
