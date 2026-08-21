"""Correlation id + access logging.

Every response carries `X-Request-ID` (echoing the caller's if it sent one), and
every log line emitted while handling that request carries the same id via the
`request_id` ContextVar. This is what makes a Flutter bug report actionable.
"""

from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, request_id_ctx

logger = get_logger("app.access")

HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(HEADER, "")
        # Don't trust an arbitrary-length client header into our log index.
        rid = incoming[:64] if incoming else uuid.uuid4().hex[:16]
        token = request_id_ctx.set(rid)
        request.state.request_id = rid

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration = (time.perf_counter() - started) * 1000
            logger.exception(
                "request_failed",
                extra={"method": request.method, "path": request.url.path,
                       "duration_ms": round(duration, 2)},
            )
            raise
        else:
            duration = (time.perf_counter() - started) * 1000
            response.headers[HEADER] = rid
            response.headers["X-Response-Time-ms"] = f"{duration:.2f}"

            # Health checks fire constantly; logging them buries the real traffic.
            if not request.url.path.startswith("/health"):
                logger.info(
                    "request",
                    extra={
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round(duration, 2),
                        "client": request.client.host if request.client else None,
                    },
                )
            return response
        finally:
            # Reset last: the access log above must still see the id.
            request_id_ctx.reset(token)
