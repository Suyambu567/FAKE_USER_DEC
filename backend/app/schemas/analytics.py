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
    above it, the model has no predictive power -- on the random-label dataset
    shipped in `data/`, it is not. Exposing both makes that impossible to hide
    behind a single impressive-looking number.

    Everything a trainer may legitimately not produce is optional. The
    cross-validation fields were required, which meant a model trained with the
    documented `--cv-folds 0` wrote metadata this model rejected, and
    `GET /api/v1/model/info` answered 500 instead of reporting the metrics it
    did have.
    """

    accuracy: float
    baseline_accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    lift_over_baseline: float = Field(
        ..., description="accuracy - baseline_accuracy. <= 0 means the model is useless."
    )

    # Present when cross-validation ran (`--cv-folds` > 1).
    cv_accuracy_mean: float | None = None
    cv_accuracy_std: float | None = None
    cv_folds: int | None = None

    # Written by ml.evaluation; absent from older artifacts.
    specificity: float | None = Field(
        None, description="True-negative rate. A false positive is a real user wrongly flagged."
    )
    pr_auc: float | None = Field(
        None, description="Average precision. The metric that matters at a low fake rate."
    )
    test_samples: int | None = None
    positive_class: str | None = None
    confusion_matrix: dict | None = Field(
        None,
        description="Labels, the 2x2 matrix, and the four cells named. Accuracy alone "
                    "cannot show whether the errors are missed fakes or flagged real users.",
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
