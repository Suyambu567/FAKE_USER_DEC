"""Application factory and lifecycle.

Startup order matters: logging first (so failures are logged), then the model,
then analytics (which needs model metadata). If the model fails to load the app
still starts and serves `/health/ready` as 503 -- a container that exits on a bad
artifact gives you a crash-loop and no diagnostic; one that reports itself unready
gets pulled from the load balancer and keeps its logs.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.v1.router import api_router
from app.api.v1.routes import health
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.logging import configure_logging, get_logger
from app.middleware.rate_limit import RateLimitMiddleware
from app.middleware.request_id import RequestIDMiddleware
from app.schemas.common import Envelope
from app.services.analytics_service import AnalyticsService
from app.services.model_service import ModelService
from app.services.profile_provider import build_provider

logger = get_logger(__name__)

DESCRIPTION = """
REST API for the fake-profile classifier, built for a Flutter mobile client.

### Response shape
Every endpoint -- success or failure -- returns the same envelope:

```json
{ "success": true, "message": "...", "data": {...}, "error": null, "request_id": "..." }
```

On failure `data` is `null` and `error.code` carries a stable machine-readable
string (`validation_error`, `rate_limited`, `model_not_ready`, ...). Branch on
`error.code`, never on the human-readable `message`.

### Model quality
`GET /api/v1/model/info` returns a `warnings` array and reports `accuracy`
alongside `baseline_accuracy`. Surface both in any UI that shows a verdict.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    started = time.perf_counter()

    model = ModelService(
        settings.model_path,
        settings.metadata_path,
        max_concurrency=settings.inference_max_concurrency,
        timeout_seconds=settings.inference_timeout_seconds,
    )
    analytics = AnalyticsService(settings.dataset_path)

    app.state.model_service = model
    app.state.analytics_service = analytics
    app.state.profile_provider = None

    try:
        model.load()
        analytics.build(model.metadata)
    except Exception:
        # Serve 503 on /health/ready rather than crash-looping with no logs.
        logger.exception("startup_model_load_failed")

    # Lookup is optional; a misconfigured provider must not stop the service
    # from serving /predict, which does not depend on it.
    try:
        app.state.profile_provider = build_provider(settings)
    except Exception:
        logger.exception("profile_provider_init_failed",
                         extra={"provider": settings.profile_provider})

    provider = app.state.profile_provider
    if provider is not None and settings.profile_provider == "mock" and settings.is_production:
        logger.warning(
            "mock_profile_provider_in_production",
            extra={"detail": "PROFILE_PROVIDER=mock fabricates profile data. "
                             "Verdicts from /lookup are meaningless."},
        )

    logger.info(
        "startup_complete",
        extra={
            "environment": settings.environment,
            "model_ready": model.is_ready,
            "profile_provider": getattr(provider, "name", "none"),
            "boot_ms": round((time.perf_counter() - started) * 1000, 1),
        },
    )

    yield  # ---- serving ----

    # Graceful shutdown: uvicorn/gunicorn stop accepting, drain in-flight
    # requests, then run this. Releasing the model frees ~0.5 MB per worker and
    # closing the provider's HTTP pool avoids a "unclosed session" warning.
    logger.info("shutdown_started")
    if provider is not None:
        await provider.aclose()
    model.unload()
    logger.info("shutdown_complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title="Fake Profile Detection API",
        description=DESCRIPTION,
        version=settings.app_version,
        lifespan=lifespan,
        root_path=settings.root_path,
        # Interactive docs are useful in prod too, but only behind the API key
        # if one is configured; they expose no data on their own.
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Middleware runs bottom-up on the request path. RequestID must be added last
    # so it runs *first* and every other layer's logs carry the correlation id.
    if settings.is_production:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    if settings.rate_limit_enabled:
        app.add_middleware(
            RateLimitMiddleware,
            requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=settings.cors_allow_credentials,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Response-Time-ms",
                        "X-RateLimit-Limit", "X-RateLimit-Remaining"],
        max_age=600,
    )
    app.add_middleware(RequestIDMiddleware)

    register_exception_handlers(app)

    app.include_router(health.router, prefix="/health")
    app.include_router(api_router, prefix="/api/v1")

    @app.get("/", tags=["meta"], response_model=Envelope[dict],
             summary="Service discovery")
    async def root() -> Envelope[dict]:
        return Envelope.ok(
            {
                "service": settings.app_name,
                "version": settings.app_version,
                "docs": "/docs",
                "openapi": "/openapi.json",
                "api_base": "/api/v1",
                "health": {"live": "/health/live", "ready": "/health/ready"},
            },
            message="Fake Profile Detection API",
        )

    return app


app = create_app()
