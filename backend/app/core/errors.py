"""Domain exceptions and the handlers that turn them into the response envelope.

The old Flask app did `flash(f'An error occurred: {str(e)}')`, which pushed raw
internal exception text (`'ColumnTransformer' object has no attribute
'_name_to_fitted_passthrough'`) to end users. Here, internal detail is logged and
the client gets a stable machine-readable `error.code` instead.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)


class AppError(Exception):
    """Base class for every error this service raises deliberately."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An internal error occurred."

    def __init__(self, message: str | None = None, *, details: Any = None) -> None:
        super().__init__(message or self.message)
        if message:
            self.message = message
        self.details = details


class ModelNotReadyError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_ready"
    message = "The prediction model is not loaded. The service is not ready."


class InferenceError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "inference_failed"
    message = "Prediction failed."


class InferenceTimeoutError(AppError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "inference_timeout"
    message = "Prediction timed out."


class ValidationError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "validation_error"
    message = "The request payload is invalid."


class AuthError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    message = "Missing or invalid API key."


class RateLimitError(AppError):
    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "rate_limited"
    message = "Too many requests. Slow down."

    def __init__(self, retry_after: int, **kw: Any) -> None:
        super().__init__(**kw)
        self.retry_after = retry_after


def _envelope(message: str, code: str, details: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "success": False,
        "message": message,
        "data": None,
        "error": {"code": code},
        "request_id": request_id_ctx.get(),
    }
    if details is not None:
        body["error"]["details"] = details
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def _app_error(_: Request, exc: AppError) -> JSONResponse:
        # 5xx is our fault and worth a stack trace; 4xx is the caller's and is not.
        if exc.status_code >= 500:
            logger.exception("app_error", extra={"code": exc.code})
        else:
            logger.warning("app_error", extra={"code": exc.code, "detail": exc.message})
        headers = {}
        if isinstance(exc, RateLimitError):
            headers["Retry-After"] = str(exc.retry_after)
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(exc.message, exc.code, exc.details),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Flatten pydantic's error list into something a Flutter client can show
        # next to the offending form field.
        details = [
            {
                "field": ".".join(str(p) for p in err["loc"] if p not in ("body", "query")),
                "message": err["msg"],
            }
            for err in exc.errors()
        ]
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=jsonable_encoder(
                _envelope("The request payload is invalid.", "validation_error", details)
            ),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found",
                405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(
            status_code=exc.status_code,
            content=_envelope(str(exc.detail), code),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, exc: Exception) -> JSONResponse:
        # Last line of defence: log everything, tell the client nothing specific.
        logger.exception("unhandled_exception", extra={"exc_type": type(exc).__name__})
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=_envelope("An internal error occurred.", "internal_error"),
        )
