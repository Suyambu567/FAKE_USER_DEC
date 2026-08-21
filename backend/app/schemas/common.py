"""The single response envelope every endpoint returns."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

from app.core.logging import request_id_ctx

T = TypeVar("T")


class ErrorDetail(BaseModel):
    code: str = Field(..., examples=["validation_error"])
    details: Any = None


class Envelope(BaseModel, Generic[T]):
    """`{success, message, data}` plus the bits a mobile client actually needs.

    `request_id` is echoed so a user can screenshot an error and support can find
    the exact log line. `error` is null on success.
    """

    success: bool = True
    message: str = "OK"
    data: T | None = None
    error: ErrorDetail | None = None
    # Populated from the ContextVar that RequestIDMiddleware sets, so every
    # handler gets it for free without threading `request` through the service
    # layer. Matches the X-Request-ID response header.
    request_id: str | None = Field(default_factory=lambda: request_id_ctx.get())

    @classmethod
    def ok(cls, data: T, message: str = "OK") -> "Envelope[T]":
        return cls(success=True, message=message, data=data)


class HealthData(BaseModel):
    status: str = Field(..., examples=["ok"])
    version: str
    environment: str
    uptime_seconds: float


class ReadinessData(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None = None
    checks: dict[str, bool]
