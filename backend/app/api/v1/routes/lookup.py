"""Username -> verdict.

`POST /predict` needs nine features. This endpoint takes only a username, fetches
those features from the configured profile provider, and runs the same prediction
path -- so there is exactly one inference code path, not two.

The response includes the fetched features under `profile`, deliberately. A
verdict with no visible inputs is unauditable: when the answer looks wrong, the
first question is always "what data did it see?".
"""

from __future__ import annotations

import re

from fastapi import APIRouter, Depends, Path, Request

from app.api.deps import get_model_service
from app.core.security import require_api_key
from app.schemas.common import Envelope
from app.schemas.lookup import LookupData
from app.schemas.predict import ProfileFeatures
from app.services.model_service import ModelService
from app.services.profile_provider import ProfileProvider, ProviderNotConfiguredError

router = APIRouter(tags=["lookup"], dependencies=[Depends(require_api_key)])

# Instagram usernames: letters, digits, periods, underscores, 1-30 chars.
# Enforced here as well as in the path regex so the rule lives in one readable place.
USERNAME_RE = re.compile(r"^[A-Za-z0-9._]{1,30}$")


def get_profile_provider(request: Request) -> ProfileProvider:
    provider: ProfileProvider | None = getattr(request.app.state, "profile_provider", None)
    if provider is None:
        raise ProviderNotConfiguredError()
    return provider


@router.get(
    "/lookup/{username}",
    response_model=Envelope[LookupData],
    summary="Classify a profile by username",
    description=(
        "Fetches the profile's features from the configured provider, then scores "
        "them with the same model `POST /predict` uses.\n\n"
        "**Requires `PROFILE_PROVIDER` to be configured.** It is `none` by default, "
        "in which case this returns `501 provider_not_configured` — Instagram's "
        "Graph API cannot look up arbitrary usernames, so a third-party data "
        "provider must be supplied. With `PROFILE_PROVIDER=mock` the endpoint "
        "returns deterministic **synthetic** data and `profile.source` is `mock`; "
        "never present that as a real verdict.\n\n"
        "The features actually used are returned under `profile` so the verdict "
        "can be audited."
    ),
    responses={
        404: {"description": "No profile found for that username."},
        501: {"description": "Username lookup is not configured."},
        502: {"description": "The upstream profile data provider failed."},
    },
)
async def lookup(
    username: str = Path(
        ...,
        min_length=1,
        max_length=30,
        pattern=r"^[A-Za-z0-9._]+$",
        description="Profile handle, without the leading '@'.",
        examples=["natgeo"],
    ),
    provider: ProfileProvider = Depends(get_profile_provider),
    model: ModelService = Depends(get_model_service),
) -> Envelope[LookupData]:
    raw = await provider.fetch(username)

    # Validate the provider's output against the same schema a direct caller must
    # satisfy. A vendor returning a negative follower count fails here, loudly,
    # instead of reaching the model.
    features = ProfileFeatures(**raw)

    labels, dists, latency, _imputed = await model.predict([features.model_dump()])
    label = labels[0]

    return Envelope.ok(
        LookupData(
            username=username,
            label=label,
            confidence=dists[0][label],
            probabilities=dists[0],
            profile=features,
            source=getattr(provider, "name", "unknown"),
            model_version=model.version,
            latency_ms=latency,
        ),
        message=f"Classified @{username}.",
    )
