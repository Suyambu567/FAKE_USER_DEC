"""Fixed-window rate limiting.

Scope caveat, stated plainly because it matters for the deployment target:
this counter lives in process memory. With N uvicorn workers the effective limit
is N x `rate_limit_requests`, and it resets on restart. That is fine up to a few
thousand users on a single box. Past that, move the counter to Redis -- the
`_Bucket` lookup is the only thing that needs to change, which is why it is
isolated behind `_hit()`.

Health endpoints are exempt so a load balancer probe can never be throttled.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.logging import get_logger, request_id_ctx

logger = get_logger(__name__)

EXEMPT_PREFIXES = ("/health", "/docs", "/redoc", "/openapi.json")


@dataclass
class _Bucket:
    count: int = 0
    window_start: float = field(default_factory=time.monotonic)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, *, requests: int, window_seconds: int) -> None:
        super().__init__(app)
        self._max = requests
        self._window = window_seconds
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    @staticmethod
    def _client_key(request: Request) -> str:
        # An API key identifies a caller far better than an IP behind CGNAT,
        # which is the normal case for mobile traffic.
        api_key = request.headers.get("X-API-Key")
        if api_key:
            return f"key:{api_key[:32]}"
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return f"ip:{forwarded.split(',')[0].strip()}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def _sweep(self, now: float) -> None:
        """Drop stale buckets so the dict cannot grow without bound."""
        if now - self._last_sweep < self._window:
            return
        cutoff = now - self._window * 2
        for key in [k for k, b in self._buckets.items() if b.window_start < cutoff]:
            self._buckets.pop(key, None)
        self._last_sweep = now

    def _hit(self, key: str) -> tuple[bool, int, int]:
        """Returns (allowed, remaining, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            self._sweep(now)
            bucket = self._buckets.get(key)
            if bucket is None or now - bucket.window_start >= self._window:
                self._buckets[key] = _Bucket(count=1, window_start=now)
                return True, self._max - 1, 0
            bucket.count += 1
            if bucket.count > self._max:
                retry = int(self._window - (now - bucket.window_start)) + 1
                return False, 0, retry
            return True, self._max - bucket.count, 0

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if request.url.path.startswith(EXEMPT_PREFIXES):
            return await call_next(request)

        key = self._client_key(request)
        allowed, remaining, retry_after = self._hit(key)

        if not allowed:
            logger.warning("rate_limited", extra={"client_key": key, "path": request.url.path})
            return JSONResponse(
                status_code=429,
                content={
                    "success": False,
                    "message": "Too many requests. Slow down.",
                    "data": None,
                    "error": {"code": "rate_limited", "details": {"retry_after": retry_after}},
                    "request_id": request_id_ctx.get(),
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(self._max),
                    "X-RateLimit-Remaining": "0",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._max)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response
