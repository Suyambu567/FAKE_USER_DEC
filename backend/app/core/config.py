"""Application configuration.

Every knob is environment-driven so the same image runs in dev, staging and prod.
Nothing here reads the filesystem at import time -- `get_settings()` is cached and
called from the lifespan handler, which keeps imports side-effect free and tests fast.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- service identity -------------------------------------------------
    app_name: str = "fake-profile-detector"
    app_version: str = "2.0.0"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = False

    # ---- network ----------------------------------------------------------
    host: str = "127.0.0.1"
    port: int = 8000
    workers: int = 2
    root_path: str = ""

    # ---- CORS (Flutter web needs this; Android/iOS do not) ----------------
    # NoDecode is required. Without it pydantic-settings tries to JSON-decode
    # any complex-typed env var *before* validators run, so the natural
    # `CORS_ORIGINS=*` or `CORS_ORIGINS=https://a.com,https://b.com` raises
    # SettingsError at import and the process cannot boot. NoDecode hands the
    # raw string to `_split_origins` below instead.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])
    cors_allow_credentials: bool = False

    # ---- artifacts --------------------------------------------------------
    # Defaults to the paper's XGBoost model (`python -m ml.train_xgb`). The older
    # RandomForest artifact is still on disk and still loadable — point
    # ARTIFACTS_DIR at `artifacts` to serve it instead; `ModelService` picks the
    # matching feature path from the metadata either way.
    artifacts_dir: Path = BASE_DIR / "artifacts" / "xgb"
    model_filename: str = "model.joblib"
    metadata_filename: str = "model_meta.json"
    dataset_path: Path | None = None

    # ---- inference tuning -------------------------------------------------
    # sklearn predict is CPU-bound and holds the GIL for part of its work, so it
    # runs in a worker thread. The semaphore stops a traffic spike from queueing
    # unbounded work and blowing up memory.
    inference_max_concurrency: int = 8
    inference_timeout_seconds: float = 5.0
    max_batch_size: int = 100

    # ---- username lookup --------------------------------------------------
    # Turns a username into the nine model features. See
    # app/services/profile_provider.py for why this is pluggable.
    #   none — disabled (default); /lookup returns 501
    #   mock — deterministic synthetic data, development only
    #   http — a third-party profile-data API (requires profile_api_url)
    profile_provider: Literal["none", "mock", "http"] = "none"
    profile_api_url: str | None = None            # must contain {username}
    profile_api_key: str | None = None
    profile_api_key_header: str = "X-API-Key"
    profile_api_timeout_seconds: float = 10.0
    profile_cache_ttl_seconds: int = 900          # 0 disables caching
    # JSON object mapping wire field -> dotted path in the provider's response.
    profile_field_map: Annotated[dict[str, str] | None, NoDecode] = None

    # ---- security ---------------------------------------------------------
    api_key: str | None = None          # when unset, auth is disabled
    api_key_header: str = "X-API-Key"
    max_bio_length: int = 2000

    # ---- rate limiting ----------------------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 60
    rate_limit_window_seconds: int = 60

    # ---- observability ----------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = True
    metrics_enabled: bool = True

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        """Accept a JSON array, a comma-separated list, or a single origin.

        NoDecode means we own the parsing, so handle every form a .env file
        realistically contains rather than making the operator guess.
        """
        if not isinstance(v, str):
            return v
        raw = v.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"CORS_ORIGINS looks like JSON but does not parse: {exc}"
                ) from exc
            if not isinstance(parsed, list):
                raise ValueError("CORS_ORIGINS JSON must be an array of strings")
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in raw.split(",") if item.strip()]

    @field_validator("dataset_path", mode="before")
    @classmethod
    def _blank_path_is_unset(cls, v: object) -> object:
        """`DATASET_PATH=` in a .env means "not configured", not `Path(".")`.

        Without this, pydantic coerces the empty string to the current
        directory, `Path(".").exists()` is True, and the analytics service logs
        an `IsADirectoryError` traceback on every single startup while the
        optional dataset panel silently stays empty.
        """
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("profile_field_map", mode="before")
    @classmethod
    def _parse_field_map(cls, v: object) -> object:
        # NoDecode again: an empty env var must mean "unset", not a JSON error.
        if v is None or v == "":
            return None
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
            except json.JSONDecodeError as exc:
                raise ValueError(f"PROFILE_FIELD_MAP is not valid JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("PROFILE_FIELD_MAP must be a JSON object")
            return {str(k): str(val) for k, val in parsed.items()}
        return v

    @property
    def model_path(self) -> Path:
        return self.artifacts_dir / self.model_filename

    @property
    def metadata_path(self) -> Path:
        return self.artifacts_dir / self.metadata_filename

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
