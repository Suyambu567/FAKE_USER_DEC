"""Liveness and readiness probes.

Split deliberately: `/health/live` answers "is the process alive" and must never
depend on the model, or a slow model load would make Kubernetes kill a container
that is merely still booting. `/health/ready` answers "can this instance serve
traffic" and is what a load balancer should gate on.

Both are exempt from auth and rate limiting.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

from app.core.config import get_settings
from app.schemas.common import Envelope, HealthData, ReadinessData

router = APIRouter(tags=["health"])
_STARTED = time.monotonic()


@router.get("/live", response_model=Envelope[HealthData], summary="Liveness probe")
async def live() -> Envelope[HealthData]:
    s = get_settings()
    return Envelope.ok(
        HealthData(
            status="ok",
            version=s.app_version,
            environment=s.environment,
            uptime_seconds=round(time.monotonic() - _STARTED, 2),
        )
    )


@router.get(
    "/ready",
    response_model=Envelope[ReadinessData],
    summary="Readiness probe",
    responses={503: {"description": "Model not loaded; do not route traffic here."}},
)
async def ready(request: Request, response: Response) -> Envelope[ReadinessData]:
    model = getattr(request.app.state, "model_service", None)
    analytics = getattr(request.app.state, "analytics_service", None)

    checks = {
        "model_loaded": bool(model and model.is_ready),
        "analytics_ready": bool(analytics and analytics.snapshot is not None),
    }
    healthy = checks["model_loaded"]  # analytics is optional

    if not healthy:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return Envelope(
        success=healthy,
        message="ready" if healthy else "not ready",
        data=ReadinessData(
            status="ready" if healthy else "not_ready",
            model_loaded=checks["model_loaded"],
            model_version=model.version if checks["model_loaded"] else None,
            checks=checks,
        ),
    )
