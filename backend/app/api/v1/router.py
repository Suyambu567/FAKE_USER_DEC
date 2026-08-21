"""v1 route aggregation.

Versioning strategy: the prefix is `/api/v1`. Breaking changes to a request or
response shape ship as `/api/v2` alongside v1, so an app store rollout (which
takes days and never reaches 100% of installs) cannot be broken by a deploy.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes import analytics, lookup, predict

api_router = APIRouter()
api_router.include_router(predict.router)
api_router.include_router(lookup.router)
api_router.include_router(analytics.router)
