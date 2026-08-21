"""Fetching a profile's features from a username.

The model needs nine features. A user only has a username. Something has to
close that gap, and *what* closes it is a business decision with cost and legal
consequences, so it is pluggable rather than hardcoded:

*   `none`  — disabled (default). `/lookup` returns 501 with an explanation.
*   `mock`  — deterministic synthetic data derived from the username. For
             development and demos only; it fabricates plausible numbers and
             never touches the network.
*   `http`  — a generic adapter for any third-party profile-data API. You supply
             the URL template, auth header and a field mapping via environment
             variables, so switching vendors is a config change, not a code change.

**On data sources.** Instagram's Graph API only exposes accounts you own or that
have authorised your app; it cannot look up an arbitrary username. Scraping
Instagram directly violates their Terms of Service and gets IP-banned quickly.
The workable route is a licensed third-party data provider. That is what the
`http` adapter is for.
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any

from app.core.errors import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---- errors ----------------------------------------------------------------


class ProviderNotConfiguredError(AppError):
    status_code = 501
    code = "provider_not_configured"
    message = (
        "Username lookup is not configured. Set PROFILE_PROVIDER to 'mock' "
        "(development) or 'http' (with PROFILE_API_URL) to enable it, or use "
        "POST /api/v1/predict with the profile features directly."
    )


class ProfileNotFoundError(AppError):
    status_code = 404
    code = "profile_not_found"
    message = "No profile found for that username."


class ProfileProviderError(AppError):
    status_code = 502
    code = "provider_error"
    message = "The upstream profile data provider failed."


# ---- interface -------------------------------------------------------------


class ProfileProvider(ABC):
    """Returns a dict keyed by the wire field names in `ProfileFeatures`."""

    name: str = "base"

    @abstractmethod
    async def fetch(self, username: str) -> dict[str, Any]:
        ...

    async def aclose(self) -> None:  # pragma: no cover - default no-op
        return None

    @staticmethod
    def _engagement_rate(likes: float, comments: float, followers: float) -> float:
        """Most providers do not return engagement rate; derive it.

        Standard definition: (avg interactions per post / followers) * 100.
        Clamped to the range the model was fitted on.
        """
        if followers <= 0:
            return 0.0
        return max(0.0, min(100.0, (likes + comments) / followers * 100.0))


# ---- mock ------------------------------------------------------------------


class MockProfileProvider(ProfileProvider):
    """Deterministic synthetic features derived from the username.

    Same username always yields the same numbers, so the endpoint is testable and
    demos are reproducible. It invents data -- never enable this in production.
    """

    name = "mock"

    _BIOS = [
        "Foodie | Reviews and recipes",
        "Photographer and content creator",
        "Digital nomad living my best life",
        "DM for collaborations or inquiries!",
        "Fitness enthusiast | Health advocate",
    ]

    async def fetch(self, username: str) -> dict[str, Any]:
        digest = hashlib.sha256(username.lower().encode()).digest()

        def pick(offset: int, lo: int, hi: int) -> int:
            span = hi - lo
            raw = int.from_bytes(digest[offset:offset + 4], "big")
            return lo + (raw % span if span else 0)

        followers = pick(0, 50, 50_000)
        likes = pick(8, 0, 1_000)
        comments = pick(12, 0, 100)

        logger.warning("mock_profile_served", extra={"username": username})
        return {
            "followers": followers,
            "following": pick(4, 10, 5_000),
            "posts": pick(16, 0, 500),
            "engagement_rate": round(self._engagement_rate(likes, comments, followers), 2),
            "avg_likes_per_post": likes,
            "avg_comments_per_post": comments,
            "verified": digest[20] < 26,          # ~10% verified
            "account_age_years": pick(24, 0, 12),
            "bio_text": self._BIOS[digest[28] % len(self._BIOS)],
        }


# ---- generic HTTP adapter --------------------------------------------------


class HttpProfileProvider(ProfileProvider):
    """Adapter for any third-party profile-data API.

    Configured entirely by environment variables so a vendor swap needs no code:

        PROFILE_API_URL=https://vendor.example/v1/instagram/{username}
        PROFILE_API_KEY=...
        PROFILE_API_KEY_HEADER=X-RapidAPI-Key
        PROFILE_FIELD_MAP={"followers":"follower_count","bio_text":"biography",...}

    The field map's values are dotted paths into the response JSON, so nested
    payloads like `{"data": {"user": {"edge_followed_by": {"count": 1}}}}` are
    reachable as `data.user.edge_followed_by.count`.
    """

    name = "http"

    # Sensible starting point; override any subset via PROFILE_FIELD_MAP.
    DEFAULT_MAP: dict[str, str] = {
        "followers": "follower_count",
        "following": "following_count",
        "posts": "media_count",
        "avg_likes_per_post": "avg_likes",
        "avg_comments_per_post": "avg_comments",
        "verified": "is_verified",
        "account_age_years": "account_age_years",
        "bio_text": "biography",
    }

    def __init__(self, url_template: str, api_key: str | None, key_header: str,
                 field_map: dict[str, str] | None, timeout: float) -> None:
        if "{username}" not in url_template:
            raise ValueError("PROFILE_API_URL must contain the '{username}' placeholder")
        self._url_template = url_template
        self._api_key = api_key
        self._key_header = key_header
        self._map = {**self.DEFAULT_MAP, **(field_map or {})}
        self._timeout = timeout
        self._client: Any = None

    async def _get_client(self):
        if self._client is None:
            import httpx  # imported lazily so the dep is optional when unused

            self._client = httpx.AsyncClient(
                timeout=self._timeout,
                # A small pool keeps us from stampeding the vendor on a burst.
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
                follow_redirects=True,
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _dig(payload: Any, path: str) -> Any:
        cur = payload
        for part in path.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            elif isinstance(cur, list) and part.isdigit():
                idx = int(part)
                cur = cur[idx] if idx < len(cur) else None
            else:
                return None
            if cur is None:
                return None
        return cur

    async def fetch(self, username: str) -> dict[str, Any]:
        import httpx

        client = await self._get_client()
        # quote() the username: it goes into a URL path we build.
        from urllib.parse import quote

        url = self._url_template.replace("{username}", quote(username, safe=""))
        headers = {self._key_header: self._api_key} if self._api_key else {}

        try:
            response = await client.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProfileProviderError("The profile data provider timed out.") from exc
        except httpx.HTTPError as exc:
            logger.exception("provider_request_failed", extra={"provider": self.name})
            raise ProfileProviderError() from exc

        if response.status_code == 404:
            raise ProfileNotFoundError()
        if response.status_code == 429:
            raise ProfileProviderError("The profile data provider rate-limited us.")
        if response.status_code >= 400:
            logger.error(
                "provider_bad_status",
                extra={"status": response.status_code, "body": response.text[:200]},
            )
            raise ProfileProviderError()

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProfileProviderError("The provider returned malformed JSON.") from exc

        return self._map_payload(payload, username)

    def _map_payload(self, payload: Any, username: str) -> dict[str, Any]:
        raw = {field: self._dig(payload, path) for field, path in self._map.items()}

        missing = [f for f in ("followers", "following", "posts") if raw.get(f) is None]
        if missing:
            logger.error(
                "provider_field_map_mismatch",
                extra={"missing": missing, "keys": list(payload)[:20] if isinstance(payload, dict) else None},
            )
            raise ProfileProviderError(
                f"The provider response is missing required fields: {', '.join(missing)}. "
                "Check PROFILE_FIELD_MAP against the vendor's response shape."
            )

        followers = float(raw["followers"] or 0)
        likes = float(raw.get("avg_likes_per_post") or 0)
        comments = float(raw.get("avg_comments_per_post") or 0)

        # Use the provider's engagement rate if it supplies one, else derive it.
        engagement = raw.get("engagement_rate")
        engagement = (
            float(engagement)
            if engagement is not None
            else self._engagement_rate(likes, comments, followers)
        )

        return {
            "followers": int(followers),
            "following": int(float(raw["following"] or 0)),
            "posts": int(float(raw["posts"] or 0)),
            "engagement_rate": round(max(0.0, min(100.0, engagement)), 2),
            "avg_likes_per_post": int(likes),
            "avg_comments_per_post": int(comments),
            "verified": bool(raw.get("verified") or False),
            "account_age_years": float(raw.get("account_age_years") or 0),
            # The model was fitted on non-empty text; a blank bio is common and
            # must not become a 422.
            "bio_text": (str(raw.get("bio_text") or "").strip() or "(no bio)")[:2000],
        }


# ---- TTL cache -------------------------------------------------------------


class CachedProfileProvider(ProfileProvider):
    """Wraps a provider with a bounded TTL cache.

    Profile stats barely move minute to minute, and every upstream call costs
    money and quota. Repeated lookups of the same username -- the common case
    when a user retries or several people check a viral account -- are served
    from memory.

    Per-process, like the rate limiter. Move to Redis when you scale past one box.
    """

    name = "cached"

    def __init__(self, inner: ProfileProvider, ttl_seconds: int, max_entries: int = 5000) -> None:
        self._inner = inner
        self._ttl = ttl_seconds
        self._max = max_entries
        self._store: dict[str, tuple[float, dict[str, Any]]] = {}

    async def fetch(self, username: str) -> dict[str, Any]:
        key = username.lower()
        now = time.monotonic()

        hit = self._store.get(key)
        if hit and now - hit[0] < self._ttl:
            logger.info("profile_cache_hit", extra={"username": key})
            return dict(hit[1])  # copy: callers must not mutate the cached entry

        value = await self._inner.fetch(username)

        if len(self._store) >= self._max:
            # Cheap eviction: drop everything expired, then the oldest if still full.
            for k in [k for k, (ts, _) in self._store.items() if now - ts >= self._ttl]:
                self._store.pop(k, None)
            if len(self._store) >= self._max:
                self._store.pop(min(self._store, key=lambda k: self._store[k][0]), None)

        self._store[key] = (now, value)
        return dict(value)

    async def aclose(self) -> None:
        await self._inner.aclose()


# ---- factory ---------------------------------------------------------------


def build_provider(settings: Any) -> ProfileProvider | None:
    """Returns the configured provider, or None when lookup is disabled."""
    kind = (settings.profile_provider or "none").lower()

    if kind == "none":
        return None

    if kind == "mock":
        inner: ProfileProvider = MockProfileProvider()
    elif kind == "http":
        if not settings.profile_api_url:
            raise ValueError("PROFILE_PROVIDER=http requires PROFILE_API_URL")
        inner = HttpProfileProvider(
            url_template=settings.profile_api_url,
            api_key=settings.profile_api_key,
            key_header=settings.profile_api_key_header,
            field_map=settings.profile_field_map,
            timeout=settings.profile_api_timeout_seconds,
        )
    else:
        raise ValueError(f"unknown PROFILE_PROVIDER: {kind!r} (expected none|mock|http)")

    if settings.profile_cache_ttl_seconds > 0:
        return CachedProfileProvider(inner, settings.profile_cache_ttl_seconds)
    return inner
