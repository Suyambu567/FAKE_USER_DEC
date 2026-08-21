"""Analytics and model metadata.

These replace the server-rendered `/analytics`, `/dashboard`, `/settings`,
`/profile` and `/word-analysis` Flask pages. A mobile client renders its own
charts, so the API returns numbers, not HTML.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_analytics_service, get_model_service
from app.core.security import require_api_key
from app.schemas.analytics import (
    AnalyticsData,
    FeatureSchemaData,
    FeatureSpec,
    ModelInfoData,
    ModelMetrics,
)
from app.schemas.common import Envelope
from app.services.analytics_service import AnalyticsService
from app.schemas.predict import ProfileFeatures
from app.services.model_service import ModelService

router = APIRouter(tags=["analytics"], dependencies=[Depends(require_api_key)])


@router.get(
    "/analytics",
    response_model=Envelope[AnalyticsData],
    summary="Dataset and model statistics",
    description=(
        "Precomputed at startup and served from memory. Safe to poll; it never "
        "touches disk per request."
    ),
)
async def analytics(
    service: AnalyticsService = Depends(get_analytics_service),
) -> Envelope[AnalyticsData]:
    return Envelope.ok(service.snapshot, message="Analytics snapshot.")


@router.get(
    "/model/info",
    response_model=Envelope[ModelInfoData],
    summary="Model provenance, metrics and quality warnings",
    description=(
        "**Always surface `warnings` in the UI.** They carry the data-quality "
        "findings recorded at training time -- including whether the model has any "
        "lift over the majority-class baseline."
    ),
)
async def model_info(
    model: ModelService = Depends(get_model_service),
) -> Envelope[ModelInfoData]:
    meta = model.metadata
    return Envelope.ok(
        ModelInfoData(
            model_version=model.version,
            algorithm=str(meta.get("algorithm", "unknown")),
            trained_at=str(meta.get("trained_at", "")),
            sklearn_version=str(meta.get("sklearn_version", "")),
            feature_count=int(meta.get("feature_count", 0)),
            classes=model.classes,
            metrics=ModelMetrics(**meta["metrics"]),
            warnings=list(meta.get("warnings", [])),
        ),
        message="Model metadata.",
    )


def _feature_specs() -> list[FeatureSpec]:
    """Derive the form contract from `ProfileFeatures` itself.

    This was a hand-written list of nine FeatureSpecs. It drifted the moment the
    schema gained the paper's profile attributes: a client that builds its form
    from this endpoint would never offer full_name / profile_picture /
    external_url, so every profile it submitted would be scored with imputed
    defaults while looking like a complete submission. Deriving it from the
    model means the endpoint cannot fall behind the schema again.
    """
    schema = ProfileFeatures.model_json_schema()
    required = set(schema.get("required", []))
    specs: list[FeatureSpec] = []
    for name, spec in schema["properties"].items():
        # Optional fields arrive as anyOf[<type>, null]; take the real branch.
        branch = spec
        if "anyOf" in spec:
            branch = next((b for b in spec["anyOf"] if b.get("type") != "null"), spec)
        kind = branch.get("type", "string")
        specs.append(FeatureSpec(
            name=name,
            type=kind,
            required=name in required,
            minimum=branch.get("minimum", branch.get("minLength")),
            maximum=branch.get("maximum", branch.get("maxLength")),
            description=spec.get("description") or branch.get("description") or "",
        ))
    return specs


_FEATURES = _feature_specs()


@router.get(
    "/features",
    response_model=Envelope[FeatureSchemaData],
    summary="Input field specification",
    description=(
        "Lets a Flutter client build and validate its form from the server "
        "contract. Derived from the request model, so it can never fall behind it."
    ),
)
async def features() -> Envelope[FeatureSchemaData]:
    return Envelope.ok(FeatureSchemaData(features=_FEATURES), message="Feature schema.")
