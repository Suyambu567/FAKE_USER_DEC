"""API contract tests.

The original project shipped zero tests, which is why `/predict` could be broken
in every environment without anyone noticing. These lock the contract a Flutter
client depends on: the envelope shape, the error codes, and the validation bounds.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
os.environ.setdefault("DATASET_PATH", str(BACKEND.parent / "website" / "dataset.csv"))
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("LOG_JSON", "false")

from app.main import create_app  # noqa: E402

VALID = {
    "followers": 5000,
    "following": 300,
    "posts": 150,
    "engagement_rate": 4.5,
    "avg_likes_per_post": 400,
    "avg_comments_per_post": 20,
    "verified": False,
    "account_age_years": 5,
    "bio_text": "Foodie | Reviews and recipes",
}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as c:
        yield c


# ---- envelope ---------------------------------------------------------------

def test_every_response_uses_the_envelope(client):
    for path in ("/", "/health/live", "/api/v1/features", "/api/v1/analytics"):
        body = client.get(path).json()
        assert set(body) >= {"success", "message", "data", "error", "request_id"}, path


def test_request_id_is_echoed_in_body_and_header(client):
    r = client.get("/api/v1/features", headers={"X-Request-ID": "trace-me-123"})
    assert r.headers["X-Request-ID"] == "trace-me-123"
    assert r.json()["request_id"] == "trace-me-123"


# ---- health -----------------------------------------------------------------

def test_liveness_never_depends_on_the_model(client):
    assert client.get("/health/live").status_code == 200


def test_readiness_reports_model_state(client):
    body = client.get("/health/ready").json()
    assert body["data"]["model_loaded"] is True
    assert body["data"]["model_version"]


# ---- prediction -------------------------------------------------------------

def test_predict_returns_label_and_calibrated_probabilities(client):
    data = client.post("/api/v1/predict", json=VALID).json()["data"]
    assert data["label"] in {"Fake", "Real"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert data["probabilities"][data["label"]] == data["confidence"]
    assert sum(data["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)


def test_arbitrary_bio_text_does_not_crash():
    """The old app label-encoded bio text against a 15-value LabelEncoder, so any
    real user bio raised ValueError -- and even a known bio then hit
    `'int' object has no attribute 'lower'` inside TF-IDF."""
    with TestClient(create_app()) as c:
        for bio in [
            "totally novel bio never seen in training",
            "<script>alert(1)</script>",
            "emoji only 🌴🔥💪",
            "x" * 2000,
            "'; DROP TABLE users; --",
        ]:
            r = c.post("/api/v1/predict", json={**VALID, "bio_text": bio})
            assert r.status_code == 200, (bio[:40], r.json())


def test_predict_does_not_mutate_across_calls(client):
    """Same input twice must give the same answer -- the old make_prediction
    mutated the caller's dict in place."""
    a = client.post("/api/v1/predict", json=VALID).json()["data"]
    b = client.post("/api/v1/predict", json=VALID).json()["data"]
    assert a["probabilities"] == b["probabilities"]


@pytest.mark.parametrize(
    "field,bad",
    [
        ("followers", -1),
        ("engagement_rate", 101),
        ("engagement_rate", -0.1),
        ("account_age_years", -3),
        ("bio_text", ""),
        ("bio_text", "   "),
        ("posts", "not-a-number"),
    ],
)
def test_validation_rejects_bad_input(client, field, bad):
    r = client.post("/api/v1/predict", json={**VALID, field: bad})
    assert r.status_code == 422
    body = r.json()
    assert body["success"] is False
    assert body["error"]["code"] == "validation_error"
    assert body["data"] is None


def test_missing_field_is_reported_by_name(client):
    payload = {k: v for k, v in VALID.items() if k != "followers"}
    body = client.post("/api/v1/predict", json=payload).json()
    assert any(d["field"] == "followers" for d in body["error"]["details"])


def test_internal_errors_never_leak_to_the_client(client):
    """No response may contain sklearn/pandas internals the way the Flask app's
    `flash(f'An error occurred: {e}')` did."""
    r = client.post("/api/v1/predict", json={**VALID, "followers": "abc"})
    blob = r.text.lower()
    for leak in ("traceback", "columntransformer", "sklearn", "site-packages", "/app/"):
        assert leak not in blob


# ---- batch ------------------------------------------------------------------

def test_batch_scores_every_item_in_order(client):
    items = [VALID, {**VALID, "followers": 10}, {**VALID, "verified": True}]
    data = client.post("/api/v1/predict/batch", json={"items": items}).json()["data"]
    assert data["count"] == 3
    assert [r["index"] for r in data["results"]] == [0, 1, 2]


def test_batch_size_is_capped(client):
    r = client.post("/api/v1/predict/batch", json={"items": [VALID] * 101})
    assert r.status_code == 422


def test_batch_rejects_empty(client):
    assert client.post("/api/v1/predict/batch", json={"items": []}).status_code == 422


# ---- metadata ---------------------------------------------------------------

def test_model_info_exposes_baseline_next_to_accuracy(client):
    m = client.get("/api/v1/model/info").json()["data"]
    assert "baseline_accuracy" in m["metrics"]
    assert "lift_over_baseline" in m["metrics"]


def test_data_quality_warnings_are_surfaced(client):
    """Every caveat recorded at training time must reach the client verbatim.

    This used to assert the specific random-label text, which only held while the
    served artifact was the RandomForest trained on `data/dataset.csv`. Asserting
    the plumbing instead keeps the guarantee that matters -- a caveat cannot be
    recorded in the artifact and then quietly dropped on the way out -- without
    pinning the test to one model's findings.
    """
    import json
    from app.core.config import get_settings

    recorded = json.loads(get_settings().metadata_path.read_text())["warnings"]
    served = client.get("/api/v1/model/info").json()["data"]["warnings"]
    assert served == recorded
    assert recorded, "the served artifact records no caveats at all -- suspicious"


def test_features_endpoint_matches_the_predict_schema(client):
    """The form contract must list every field /predict accepts, and mark which
    ones are optional -- a client that omits an optional field gets a prediction
    made partly from training defaults, and it can only know that if the
    contract told it the field existed."""
    from app.schemas.predict import ProfileFeatures

    specs = {f["name"]: f for f in client.get("/api/v1/features").json()["data"]["features"]}
    assert set(specs) == set(ProfileFeatures.model_fields)
    assert set(VALID) <= set(specs), "the original required fields must still be listed"
    assert all(specs[name]["required"] for name in VALID)
    assert not specs["full_name"]["required"]


# ---- auth -------------------------------------------------------------------

def test_api_key_is_enforced_when_configured(monkeypatch):
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("API_KEY", "secret-key")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    try:
        with TestClient(create_app()) as c:
            assert c.post("/api/v1/predict", json=VALID).status_code == 401
            assert c.post("/api/v1/predict", json=VALID,
                          headers={"X-API-Key": "wrong"}).status_code == 401
            assert c.post("/api/v1/predict", json=VALID,
                          headers={"X-API-Key": "secret-key"}).status_code == 200
            # Probes must stay reachable for the load balancer.
            assert c.get("/health/ready").status_code == 200
    finally:
        get_settings.cache_clear()


# ---- configuration ----------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("*", ["*"]),
        ("https://a.com", ["https://a.com"]),
        ("https://a.com,https://b.com", ["https://a.com", "https://b.com"]),
        ("https://a.com, https://b.com ", ["https://a.com", "https://b.com"]),
        ('["https://a.com"]', ["https://a.com"]),          # JSON form still works
    ],
)
def test_cors_origins_parses_from_env(monkeypatch, raw, expected):
    """Regression: pydantic-settings JSON-decodes complex types from env before
    validators run, so a bare `CORS_ORIGINS=*` used to raise SettingsError at
    import and the container crash-looped with 'Worker failed to boot'."""
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("CORS_ORIGINS", raw)
    try:
        assert get_settings().cors_origins == expected
    finally:
        get_settings.cache_clear()


def test_app_boots_with_a_full_env_file(monkeypatch):
    """Boot with every value the shipped .env.example sets -- the container did
    not, because one of them could not be parsed."""
    from app.core.config import get_settings

    get_settings.cache_clear()
    for key, value in {
        "ENVIRONMENT": "production", "DEBUG": "false",
        "HOST": "0.0.0.0", "PORT": "8000", "WORKERS": "2",
        "CORS_ORIGINS": "*", "CORS_ALLOW_CREDENTIALS": "false",
        "INFERENCE_MAX_CONCURRENCY": "8", "INFERENCE_TIMEOUT_SECONDS": "5.0",
        "MAX_BATCH_SIZE": "100", "API_KEY": "", "API_KEY_HEADER": "X-API-Key",
        "RATE_LIMIT_ENABLED": "false", "RATE_LIMIT_REQUESTS": "60",
        "RATE_LIMIT_WINDOW_SECONDS": "60", "LOG_LEVEL": "INFO",
        "LOG_JSON": "true", "METRICS_ENABLED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    try:
        with TestClient(create_app()) as c:
            assert c.get("/health/ready").status_code == 200
    finally:
        get_settings.cache_clear()


# ---- rate limiting ----------------------------------------------------------

def test_rate_limit_returns_429_with_retry_after(monkeypatch):
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("RATE_LIMIT_REQUESTS", "3")
    monkeypatch.setenv("API_KEY", "")
    try:
        with TestClient(create_app()) as c:
            codes = [c.get("/api/v1/features").status_code for _ in range(6)]
            assert 429 in codes
            r = c.get("/api/v1/features")
            assert r.status_code == 429
            assert int(r.headers["Retry-After"]) > 0
            assert r.json()["error"]["code"] == "rate_limited"
            # Health must never be throttled.
            assert c.get("/health/live").status_code == 200
    finally:
        get_settings.cache_clear()


# ---- username lookup --------------------------------------------------------

def test_lookup_is_disabled_by_default(client):
    """PROFILE_PROVIDER defaults to none -- the endpoint must say why, not 500."""
    r = client.get("/api/v1/lookup/natgeo")
    assert r.status_code == 501
    body = r.json()
    assert body["error"]["code"] == "provider_not_configured"
    assert "PROFILE_PROVIDER" in body["message"]


def _mock_client(monkeypatch):
    from app.core.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setenv("PROFILE_PROVIDER", "mock")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    monkeypatch.setenv("API_KEY", "")
    return TestClient(create_app())


def test_lookup_returns_a_verdict_and_the_features_it_used(monkeypatch):
    from app.core.config import get_settings

    try:
        with _mock_client(monkeypatch) as c:
            data = c.get("/api/v1/lookup/natgeo").json()["data"]
            assert data["username"] == "natgeo"
            assert data["label"] in {"Fake", "Real"}
            # The verdict must be auditable: every input the model scored comes
            # back with it. A superset is correct -- the schema also carries the
            # paper's optional profile attributes (full_name, profile_picture,
            # external_url, ...), which a provider may or may not supply.
            assert set(VALID) <= set(data["profile"])
            assert data["source"] in {"mock", "cached"}
    finally:
        get_settings.cache_clear()


def test_lookup_is_deterministic_and_cached(monkeypatch):
    from app.core.config import get_settings

    try:
        with _mock_client(monkeypatch) as c:
            a = c.get("/api/v1/lookup/natgeo").json()["data"]
            b = c.get("/api/v1/lookup/NatGeo").json()["data"]   # cache key is lowercased
            assert a["profile"] == b["profile"]
            assert a["label"] == b["label"]
    finally:
        get_settings.cache_clear()


@pytest.mark.parametrize("bad", ["with space", "has/slash", "x" * 31, "emoji🌴"])
def test_lookup_rejects_invalid_usernames(monkeypatch, bad):
    from app.core.config import get_settings

    try:
        with _mock_client(monkeypatch) as c:
            assert c.get(f"/api/v1/lookup/{bad}").status_code in (404, 422)
    finally:
        get_settings.cache_clear()


def test_http_provider_maps_nested_payloads_and_derives_engagement():
    """The field map takes dotted paths, and engagement rate is computed when the
    vendor does not supply it."""
    from app.services.profile_provider import HttpProfileProvider

    p = HttpProfileProvider(
        url_template="https://vendor.example/{username}",
        api_key=None, key_header="X-API-Key",
        field_map={
            "followers": "data.user.edge_followed_by.count",
            "following": "data.user.edge_follow.count",
            "posts": "data.user.media_count",
            "bio_text": "data.user.biography",
            "verified": "data.user.is_verified",
            "avg_likes_per_post": "stats.avg_likes",
            "avg_comments_per_post": "stats.avg_comments",
        },
        timeout=5.0,
    )
    payload = {
        "data": {"user": {
            "edge_followed_by": {"count": 1000},
            "edge_follow": {"count": 200},
            "media_count": 50,
            "biography": "hello world",
            "is_verified": True,
        }},
        "stats": {"avg_likes": 40, "avg_comments": 10},
    }
    out = p._map_payload(payload, "someone")
    assert out["followers"] == 1000 and out["following"] == 200
    assert out["verified"] is True
    assert out["engagement_rate"] == 5.0          # (40+10)/1000*100
    assert out["bio_text"] == "hello world"


def test_http_provider_reports_a_field_map_mismatch_clearly():
    from app.services.profile_provider import HttpProfileProvider, ProfileProviderError

    p = HttpProfileProvider("https://v/{username}", None, "X-API-Key", None, 5.0)
    with pytest.raises(ProfileProviderError) as exc:
        p._map_payload({"totally": "different"}, "someone")
    assert "PROFILE_FIELD_MAP" in str(exc.value)


def test_blank_bio_does_not_become_a_validation_error():
    """Empty bios are common upstream; the schema requires non-empty text."""
    from app.services.profile_provider import HttpProfileProvider

    p = HttpProfileProvider("https://v/{username}", None, "X-API-Key", None, 5.0)
    out = p._map_payload(
        {"follower_count": 10, "following_count": 5, "media_count": 1, "biography": ""},
        "someone",
    )
    assert out["bio_text"] == "(no bio)"
