"""API-key authentication.

Deliberately simple: this service holds no user data and has no sessions, so a
shared key checked in constant time is the right amount of auth. It is *optional*
-- leave `API_KEY` unset in development and the dependency is a no-op.

If per-user identity is ever needed (saved scans, quotas, billing), replace this
with OAuth2/JWT via `fastapi.security.OAuth2PasswordBearer`; every route already
depends on `require_api_key`, so the swap is one file.

Note on mobile: an API key shipped inside a Flutter binary is extractable. It is
a throttling and attribution control, not a secret. Anything stronger needs a
per-user token minted server-side.
"""

from __future__ import annotations

import secrets

from fastapi import Request

from app.core.config import Settings, get_settings
from app.core.errors import AuthError


async def require_api_key(request: Request) -> None:
    settings: Settings = get_settings()
    if not settings.api_key:
        return  # auth disabled

    provided = request.headers.get(settings.api_key_header, "")
    # compare_digest avoids leaking key length/prefix through timing.
    if not provided or not secrets.compare_digest(provided, settings.api_key):
        raise AuthError()
