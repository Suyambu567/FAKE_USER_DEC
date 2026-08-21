"""Prediction endpoints -- the core of the service."""

from __future__ import annotations

from fastapi import APIRouter, Depends, status

from app.api.deps import get_model_service
from app.core.security import require_api_key
from app.schemas.common import Envelope
from app.schemas.predict import (
    BatchPredictionData,
    BatchPredictionItem,
    BatchPredictRequest,
    PredictionData,
    ProfileFeatures,
)
from app.services.model_service import ModelService

router = APIRouter(tags=["prediction"], dependencies=[Depends(require_api_key)])

_ERRORS = {
    401: {"description": "Missing or invalid API key."},
    422: {"description": "Payload failed validation."},
    429: {"description": "Rate limit exceeded."},
    503: {"description": "Model not loaded."},
    504: {"description": "Inference timed out."},
}


@router.post(
    "/predict",
    response_model=Envelope[PredictionData],
    status_code=status.HTTP_200_OK,
    responses=_ERRORS,
    summary="Classify a single profile",
    description=(
        "Returns the predicted label with a calibrated-as-trained probability for "
        "every class.\n\n"
        "The paper's Table 1 profile attributes (`full_name`, `profile_picture`, "
        "`external_url`, `language`) are optional so an older client keeps working. "
        "Anything omitted is filled with the training-set default and named in "
        "`imputed_fields` — a non-empty list means part of the verdict rests on "
        "assumptions rather than on the profile.\n\n"
        "**Read the caveats on `GET /api/v1/model/info` before acting on this "
        "output.** That endpoint reports accuracy next to the majority-class "
        "baseline, and carries the warnings recorded at training time — including "
        "which Table 1 attributes this dataset could not supply."
    ),
)
async def predict_one(
    payload: ProfileFeatures,
    model: ModelService = Depends(get_model_service),
) -> Envelope[PredictionData]:
    labels, dists, latency, imputed = await model.predict([payload.model_dump()])
    label = labels[0]
    return Envelope.ok(
        PredictionData(
            label=label,
            confidence=dists[0][label],
            probabilities=dists[0],
            model_version=model.version,
            latency_ms=latency,
            imputed_fields=imputed[0],
        ),
        message="Prediction complete.",
    )


@router.post(
    "/predict/batch",
    response_model=Envelope[BatchPredictionData],
    responses=_ERRORS,
    summary="Classify up to 100 profiles in one call",
    description=(
        "Prefer this over N single calls. One vectorised `predict_proba` over the "
        "whole batch costs barely more than a single row, because the per-call "
        "pandas/numpy setup dominates single-row inference."
    ),
)
async def predict_batch(
    payload: BatchPredictRequest,
    model: ModelService = Depends(get_model_service),
) -> Envelope[BatchPredictionData]:
    items = [item.model_dump() for item in payload.items]
    labels, dists, latency, imputed = await model.predict(items)

    results = [
        BatchPredictionItem(
            index=i,
            label=label,
            confidence=dist[label],
            probabilities=dist,
            imputed_fields=fields,
        )
        for i, (label, dist, fields) in enumerate(zip(labels, dists, imputed))
    ]
    return Envelope.ok(
        BatchPredictionData(
            results=results,
            count=len(results),
            model_version=model.version,
            latency_ms=latency,
        ),
        message=f"Scored {len(results)} profiles.",
    )
