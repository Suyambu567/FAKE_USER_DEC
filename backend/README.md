# Fake Profile Detection API

Production FastAPI backend for the fake-profile classifier, built to be consumed by a Flutter
mobile client (Android / iOS / Web).

It replaces the Flask app in `../website/`. See `../docs/AUDIT.md` for why, and
`../docs/FLUTTER_INTEGRATION.md` for the client-side guide.

> **Read this before shipping.** The dataset in this repository has randomly assigned labels
> (`data/app.py` generates `Account Type` with `np.random.choice`). Measured held-out accuracy is
> **0.523 against a 0.508 majority-class baseline** — no predictive power. The API is production
> ready; *the model is not*. `GET /api/v1/model/info` returns this as a machine-readable
> `warnings` array so a client cannot present a coin flip as a verdict. Retrain on labelled
> real-world data before using any output to action a real account.

---

## Layout

```
backend/
├── app/
│   ├── main.py                     # app factory, lifespan, middleware wiring
│   ├── api/
│   │   ├── deps.py                 # DI providers (model, analytics)
│   │   └── v1/
│   │       ├── router.py           # v1 aggregation
│   │       └── routes/
│   │           ├── health.py       # /health/live, /health/ready
│   │           ├── predict.py      # /predict, /predict/batch
│   │           └── analytics.py    # /analytics, /model/info, /features
│   ├── core/
│   │   ├── config.py               # pydantic-settings, env-driven
│   │   ├── logging.py              # JSON logs + request-id ContextVar
│   │   ├── errors.py               # domain errors + handlers
│   │   └── security.py             # API-key auth
│   ├── middleware/
│   │   ├── request_id.py           # correlation id + access log
│   │   └── rate_limit.py           # fixed-window limiter
│   ├── schemas/                    # pydantic request/response models
│   │   ├── common.py               # the response envelope
│   │   ├── predict.py
│   │   └── analytics.py
│   ├── services/
│   │   ├── model_service.py        # load-once, threadpool inference
│   │   └── analytics_service.py    # startup-cached dataset stats
│   └── utils/
├── ml/
│   └── train.py                    # training + honest evaluation + metadata
├── artifacts/
│   ├── model.joblib                # 0.53 MB
│   └── model_meta.json             # version, metrics, warnings
├── tests/
│   └── test_api.py                 # 30 contract tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── start.sh
```

---

## Quick start (local)

```bash
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt

# Train once; artifacts/ is populated.
.venv/bin/python -m ml.train --dataset ../website/dataset.csv --out artifacts

cp .env.example .env          # then edit
DATASET_PATH=../website/dataset.csv \
  .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

* Swagger UI — <http://127.0.0.1:8000/docs>
* ReDoc — <http://127.0.0.1:8000/redoc>
* OpenAPI JSON — <http://127.0.0.1:8000/openapi.json>

Tests:

```bash
.venv/bin/python -m pytest tests/ -q      # 30 passed
```

---

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| `GET`  | `/` | no | Service discovery |
| `GET`  | `/health/live` | no | Liveness — never touches the model |
| `GET`  | `/health/ready` | no | Readiness — 503 when the model is not loaded |
| `POST` | `/api/v1/predict` | yes* | Classify one profile |
| `POST` | `/api/v1/predict/batch` | yes* | Classify up to 100 profiles |
| `GET`  | `/api/v1/analytics` | yes* | Dataset + model statistics (cached) |
| `GET`  | `/api/v1/model/info` | yes* | Provenance, metrics, **quality warnings** |
| `GET`  | `/api/v1/features` | yes* | Input field spec for dynamic forms |

\* Auth applies only when `API_KEY` is set. Health endpoints are always open and are exempt
from rate limiting so a load-balancer probe can never be throttled.

Every response uses one envelope:

```json
{ "success": true, "message": "...", "data": { }, "error": null, "request_id": "a1b2c3" }
```

On failure `data` is `null` and `error.code` is a stable string. **Branch on `error.code`,
never on `message`.**

| `error.code` | HTTP | Meaning |
|---|---|---|
| `validation_error` | 422 | Payload rejected; `error.details[]` has `{field, message}` |
| `unauthorized` | 401 | Missing/invalid API key |
| `rate_limited` | 429 | Back off; honour the `Retry-After` header |
| `model_not_ready` | 503 | Instance has no model; retry another instance |
| `inference_timeout` | 504 | Prediction exceeded the timeout |
| `internal_error` | 500 | Logged server-side with `request_id` |

---

## Deployment

### Docker

```bash
cd backend
cp .env.example .env
# Pick a free host port first -- 8000 is contended on almost any shared machine.
ss -ltn | grep ':8200 ' || echo "8200 is free"
docker compose up -d --build
docker compose logs -f api
curl -fsS http://127.0.0.1:8200/health/ready
```

Stop it with `docker compose down` (from this directory).

**Two things this compose file gets right, and why they matter on a shared host:**

1. **`name: fake-profile-detector` is set explicitly.** Compose otherwise derives the project
   name from the directory — which here is `backend`, a name half the world uses. Without the
   explicit name, this stack silently merges into any other `backend` project on the same
   daemon, Compose reports the other project's containers as "orphans", and
   `docker compose down --remove-orphans` from this directory **deletes them**. Never remove
   that line, and never pass `--remove-orphans` here.

2. **The host port is `${API_PORT:-8200}`, bound to loopback.** Port 8000 is the single most
   contended port on a shared box. Set `API_PORT` in `.env` if 8200 is taken. Terminate TLS in
   a reverse proxy in front; do not publish on `0.0.0.0` on a host with a public IP.

If `docker compose up` reports `port is already allocated`, another container owns that port —
find it with `docker ps --format '{{.Names}}\t{{.Ports}}' | grep 8200` and pick a different
`API_PORT` rather than stopping whatever is there.

Minimal nginx:

```nginx
server {
    listen 443 ssl http2;
    server_name api.example.com;
    ssl_certificate     /etc/letsencrypt/live/api.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 60s;
    }
}
```

### systemd (no Docker)

```ini
# /etc/systemd/system/fpd-api.service
[Unit]
Description=Fake Profile Detection API
After=network.target

[Service]
Type=exec
User=appuser
WorkingDirectory=/opt/fpd/backend
EnvironmentFile=/opt/fpd/backend/.env
ExecStart=/opt/fpd/backend/.venv/bin/gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker --workers 8 \
  --bind 127.0.0.1:8000 --graceful-timeout 30 --max-requests 2000
Restart=always
RestartSec=5
KillSignal=SIGTERM
TimeoutStopSec=45

[Install]
WantedBy=multi-user.target
```

### Sizing

Measured on this 80-core host, artifact loaded, `RATE_LIMIT_ENABLED=false`:

| Config | Concurrency | Throughput | p50 | p95 |
|---|---|---|---|---|
| 2 workers, single | 16 | 36 rps | 434 ms | 604 ms |
| 8 workers, single | 32 | 94 rps | 209 ms | 940 ms |
| 8 workers, single | 128 | 117 rps | 656 ms | 2737 ms |
| 8 workers, **batch(100)** | 32 | **8 639 profiles/s** | 308 ms | 682 ms |

Inference is CPU-bound, so throughput scales with worker count, not with async concurrency.
Start with `WORKERS = cpu_cores`, set `OMP_NUM_THREADS=1` (the Dockerfile already does) to stop
BLAS oversubscription, and **use `/predict/batch` wherever the client scores more than one
profile** — it is ~74x more efficient per profile.

### Graceful shutdown

`start.sh` `exec`s gunicorn so it becomes PID 1 and receives SIGTERM directly.
`--graceful-timeout 30` drains in-flight requests; the lifespan handler then releases the model
and logs `shutdown_complete`. Rolling deploys are invisible to clients.

---

## Configuration

Every setting is an environment variable — see `.env.example` for the annotated list.
The ones that matter in production:

| Variable | Default | Note |
|---|---|---|
| `ENVIRONMENT` | `development` | Set to `production` |
| `WORKERS` | `2` | Set to core count |
| `API_KEY` | *(unset)* | Unset ⇒ auth disabled. `python -c "import secrets;print(secrets.token_urlsafe(32))"` |
| `CORS_ORIGINS` | `*` | Narrow to real origins for Flutter Web |
| `RATE_LIMIT_REQUESTS` | `60` | Per window, **per worker** — see below |
| `INFERENCE_MAX_CONCURRENCY` | `8` | Threadpool cap per worker |
| `DATASET_PATH` | *(unset)* | Optional; enables histogram in `/analytics` |

**Known limitation:** the rate limiter counts in process memory, so the effective limit is
`WORKERS x RATE_LIMIT_REQUESTS` and it resets on restart. That is fine for a single box. Move
the counter to Redis before scaling horizontally — only `RateLimitMiddleware._hit()` changes.

---

## Retraining

```bash
.venv/bin/python -m ml.train \
  --dataset /path/to/labelled.csv \
  --out artifacts \
  --n-estimators 200 --max-depth 12
```

The dataset needs these columns: `Followers`, `Following`, `Posts`, `Engagement Rate (%)`,
`Avg Likes per Post`, `Avg Comments per Post`, `Verified`, `Account Age (Years)`, `Bio Text`,
`Account Type`.

The script audits the data before training and writes any problems into
`artifacts/model_meta.json → warnings`, which the API serves verbatim on
`GET /api/v1/model/info`. It flags:

* a `Bio Text` column with too few distinct values to carry signal,
* numeric features that do not separate the classes (the random-label case),
* a model whose accuracy does not clear the majority-class baseline.

`model_meta.json` records the exact scikit-learn version used. The API **refuses to start
serving** if the running version differs, instead of loading the pickle and then failing on
every prediction — which is precisely how the original Flask app broke.
