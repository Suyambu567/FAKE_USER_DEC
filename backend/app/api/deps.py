"""Dependency providers.

Services are created once in the lifespan handler and stashed on `app.state`.
These accessors pull them off the request so routes stay free of globals -- the
original app reached for module-level `loaded_model` with a `global` statement
inside a view, which is exactly what makes a Flask app untestable.
"""

from __future__ import annotations

from fastapi import Request

from app.core.errors import ModelNotReadyError
from app.services.analytics_service import AnalyticsService
from app.services.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    service: ModelService | None = getattr(request.app.state, "model_service", None)
    if service is None or not service.is_ready:
        raise ModelNotReadyError()
    return service


def get_analytics_service(request: Request) -> AnalyticsService:
    service: AnalyticsService | None = getattr(request.app.state, "analytics_service", None)
    if service is None or service.snapshot is None:
        raise ModelNotReadyError("Analytics are not available: the model is not loaded.")
    return service
