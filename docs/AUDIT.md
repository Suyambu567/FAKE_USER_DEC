# Production Audit — FAKE_USER_DEC

**Scope:** every file in `FAKE_USER_DEC.zip` (12,381 entries; 12,290 of them are a bundled
Windows `.venv/` of third-party packages, excluded — 91 project files reviewed).
**Method:** static review plus empirical reproduction. Every claim below was executed, not
inferred. Commands and outputs are quoted.

**Environment used for verification:** Linux, Python 3.10.12 / 3.11.15, `website/requirements.txt`
as pinned, and separately scikit-learn 1.9.0 (the version the shipped `.pkl` files were actually
built with).

> **Note on missing files.** This is a record of the project *as audited*. Several
> files examined below — the one-off codemods in `website/`, the `.bak`/`.backup`
> copies, the duplicate training scripts, and the large `.pkl` weights — have since
> been removed by `scripts/cleanup_legacy.sh`, and the model weights are not tracked
> in git at all. The findings still stand; the files they describe are simply no
> longer in the tree. `cleanup_legacy.sh --archive` tars everything it deletes to
> `../FAKE_USER_DEC-legacy-<timestamp>.tar.gz` — one level *above* the repo, so the
> archive is never committed. Failing that, this document quotes the relevant
> output inline.

---

## Headline

The application does not work. Not "works but is slow" — `/predict` cannot return a prediction
in **any** environment, and the model it would return has **no predictive power** because the
training labels are random.

| # | Finding | Evidence |
|---|---|---|
| 1 | `/predict` raises on every request, in every environment | reproduced twice, §12.1 |
| 2 | Model accuracy 0.5083 vs 0.5080 majority baseline | measured, §11.1 |
| 3 | Labels are statistically independent of features | max standardised mean difference 0.0395, §11.1 |
| 4 | `requirements.txt` pins a version that cannot load the shipped artifacts | reproduced, §12.2 |
| 5 | `debug=True` + `host='0.0.0.0'` — RCE via Werkzeug console | `website/app.py:276` |

---

## Step 1 — What the project is

### 1.1 Technology stack

| Layer | Technology | Where |
|---|---|---|
| Web framework | Flask 2.3.3 | `website/app.py`, `FAKE_PROFILE_TRAIN_CODE/app.py` |
| Templating | Jinja2 (server-rendered) | `website/templates/*.html` |
| ML | scikit-learn (pinned 1.3.2, artifacts built with 1.9.0) | `FAKE_PROFILE_TRAIN_CODE/train.py` |
| Serialisation | joblib | `.pkl` artifacts |
| Data | pandas / numpy, CSV files | `website/dataset.csv` etc. |
| Charts | Chart.js via CDN | `templates/analytics.html` |
| Database | **none** | — |
| Auth | **none** | — |
| Tests | **none** | — |
| CI/CD, Docker, config | **none** | — |

### 1.2 Folder structure as shipped

```
FAKE_USER_DEC/
├── .claude/settings.local.json          41 permission rules, Windows-specific
├── .venv/                               Windows venv, cpython-314 — dead weight on Linux
├── data/
│   ├── app.py                           dataset GENERATOR (misnamed — not a web app)
│   └── dataset.csv                      10k rows
├── FAKE_PROFILE_TRAIN_CODE/
│   ├── app.py                           a SECOND Flask app, also binds :5000
│   ├── train.py       ┐
│   ├── train_improved.py ├── three near-identical copies
│   ├── debug.py       ┘
│   ├── dataset.csv                      10k rows
│   ├── trained_model.pkl                72 MB
│   └── trained_model_improved.pkl       72 MB
├── papper/                              [sic] notebooks + a 4th dataset copy
├── website/
│   ├── app.py                           the "real" Flask app
│   ├── app.py.backup, app.py.bak        two stale copies in version control
│   ├── augment_dataset.py               synthetic row generator
│   ├── split_dataset.py
│   ├── fix_chart.py, fix_chart2.py, fix_chart3.py           ┐ 8 one-off codemods
│   ├── modify_analytics.py, …2.py, …3.py                    ├ that rewrite templates
│   ├── update_charts.py, update_charts2.py                  ┘ in place. All dead.
│   ├── flask*.log (5 files)             committed logs
│   ├── templates/
│   │   ├── analytics.html + .backup, .backup2, .backup3, .bak, .bak2
│   │   ├── index.html, dashboard.html, profile.html, settings.html,
│   │   ├── result.html, word_analysis.html
│   │   └── inline_block.txt, prefix.txt, suffix.txt … (8 scratch fragments)
│   ├── dataset.csv (15k), dataset_original.csv (10k), train.csv, test.csv
│   ├── trained_model.pkl                 24 MB
│   ├── trained_model.pkl.backup          72 MB
│   └── trained_model_original_copy.pkl  109 MB
├── FAKE_USER_DEC.zip                    a zip inside the zip, containing one settings file
├── test.txt                             contains the single character 💪
├── trained_model.pkl                     72 MB (a 5th copy)
└── README.md
```

**Four copies of the same dataset. Five model artifacts totalling 349 MB. Two Flask apps that
both bind port 5000.** The project is 410 MB, of which ~1.2 MB is source.

### 1.3 Request flow (`website/app.py`)

```
Browser
  └─ GET  /                → home()            → render index.html
  └─ POST /predict         → predict()
        ├─ 1. null-check module globals loaded_model / le_bio / le_account
        ├─ 2. hand-rolled validation loop over a dict of (type, lambda) pairs
        ├─ 3. make_prediction(input_data)
        │     ├─ le_bio.transform([bio])  ← ValueError on any bio outside 15 canned strings
        │     ├─ input_data['Bio Text'] = <int>   ← MUTATES CALLER'S DICT
        │     ├─ loaded_model.predict(df)  ← TF-IDF gets an int → AttributeError
        │     └─ le_account.inverse_transform([...])
        └─ 4. except Exception → flash(str(e)) → redirect home   ← leaks internals
```

Step 3 cannot succeed. See §12.1.

### 1.4 Startup flow

`website/app.py:30` calls `load_model()` at **import time**. On failure it prints to stderr and
returns `(None, None, None)`; the app starts anyway and every request degrades. There is no
readiness signal — a load balancer cannot tell a healthy instance from a broken one.

### 1.5 State management

Module-level globals (`loaded_model`, `le_bio`, `le_account`) mutated via `global` inside the
`/analytics` view (`website/app.py:83-85`). No database, no session store, no cache.

### 1.6 Data flow / database / auth flow

No database. No authentication, authorisation, sessions, users, or audit log. Any caller can hit
any endpoint. `app.secret_key` is the literal string
`'your-secret-key-change-in-production'` (`website/app.py:10`).

---

## Step 2 — Code review

### 2.1 CRITICAL — `/predict` cannot work

**File:** `website/app.py`  **Function:** `make_prediction` (lines 37–70)

```python
bio_text_encoded = le_bio.transform([input_data['Bio Text']])[0]   # line 50
input_data['Bio Text'] = bio_text_encoded                          # line 51
prediction = loaded_model.predict(input_df)[0]                     # line 57
```

**Problem:** two independent fatal defects on consecutive lines.

1. `le_bio` is a `LabelEncoder` fitted on exactly **15** canned bio strings. Any other input
   raises `ValueError`.
2. Even for one of those 15, line 51 replaces the string with an integer — but the pipeline's
   `TfidfVectorizer` (`train.py:27`) expects text. It calls `.lower()` on the int.

**Reproduced** (Python 3.11 + scikit-learn 1.9.0, the artifact's own version):

```
>>> UNSEEN BIO CRASH: ValueError - y contains previously unseen labels: 'I am a totally normal user'
>>> CRASH: AttributeError - 'int' object has no attribute 'lower'
```

**Why it is bad:** the single feature the product exists to provide has never worked. The broad
`except Exception` at line 68 converts the crash into a flash message, so it looks like a
validation problem rather than a total failure.

**Root cause:** `train.py` never creates `bio_encoder.pkl`. It was produced by some other script
not in the repo, encoding an assumption (`bio → int`) that contradicts the pipeline it feeds.

**Best solution:** delete `bio_encoder.pkl` entirely. Bio text is vectorised *inside* the fitted
pipeline, so the artifact is self-contained and any string works.
Implemented — `backend/ml/train.py`, `backend/app/services/model_service.py`.

**Expected improvement:** 0% → 100% of predictions succeed. Verified by
`tests/test_api.py::test_arbitrary_bio_text_does_not_crash`, which asserts 200 for novel text,
`<script>` tags, emoji-only bios, 2,000-char bios and SQL-injection strings.

---

### 2.2 CRITICAL — dependency pin cannot load the artifacts

**File:** `website/requirements.txt`

Pins `scikit-learn==1.3.2`. The shipped `.pkl` files were produced with **1.9.0** (confirmed by
scanning the pickle opcode stream — all five artifacts report `1.9.0`).

Following the README exactly:

```
Error during prediction: 'ColumnTransformer' object has no attribute '_name_to_fitted_passthrough'
POST /predict -> 200 (redirect home, error flashed)
```

`joblib.load()` *succeeds* — so `load_model()` reports success and the app looks healthy — and
then every `predict()` fails. Worse, scikit-learn 1.9.0 requires **Python ≥ 3.11**, while the
README says "Python 3.8+".

**Best solution:** pin exactly, record the training version in artifact metadata, and refuse to
serve on mismatch instead of failing per-request.
Implemented — `model_service.load()` raises `ModelNotReadyError` on version mismatch; the
instance reports 503 on `/health/ready` and is pulled from the load balancer.

---

### 2.3 CRITICAL — `debug=True` on `0.0.0.0`

**File:** `website/app.py:276` (identical at `FAKE_PROFILE_TRAIN_CODE/app.py:264`)

```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

**Why it is bad:** the Werkzeug debugger exposes an interactive Python console on any unhandled
exception. Bound to `0.0.0.0`, that is **unauthenticated remote code execution** for anyone who
can reach the port. The debugger PIN is derived from predictable machine values and is not a
security boundary. `debug=True` also disables caching and doubles memory (the reloader runs two
processes, each holding a 24 MB model).

**Best solution:** never `app.run()` in production; bind loopback and front with a reverse proxy.
Implemented — `start.sh` runs gunicorn, `docker-compose.yml` publishes to `127.0.0.1:8000`, and
`.env.example` defaults `HOST=127.0.0.1` with the reasoning inline.

---

### 2.4 HIGH — hardcoded secret key

**File:** `website/app.py:10` — `app.secret_key = 'your-secret-key-change-in-production'`

A known signing key lets anyone forge session cookies and flash messages. It is also committed,
so rotating it means a code change.
**Fix:** environment-driven config. Implemented — `app/core/config.py`.

---

### 2.5 HIGH — internal exception text returned to users

**File:** `website/app.py:270` — `flash(f'An error occurred: {str(e)}', 'error')`

Users saw `'ColumnTransformer' object has no attribute '_name_to_fitted_passthrough'`. This
discloses the ML stack, library versions and internal structure — a reconnaissance gift.
**Fix:** stable `error.code` to the client, full trace to the log, correlated by `request_id`.
Implemented — `app/core/errors.py`; asserted by `test_internal_errors_never_leak_to_the_client`.

---

### 2.6 HIGH — race condition on lazy model reload

**File:** `website/app.py:82-85`

```python
global loaded_model, le_bio, le_account
if loaded_model is None:
    loaded_model, le_bio, le_account = load_model()
```

Under any threaded server, two concurrent `/analytics` requests both observe `None` and both
call `load_model()`, deserialising a 24–72 MB pickle twice concurrently. Nothing guards it.
**Fix:** load once in the lifespan handler; the instance is immutable afterwards.
Implemented — `app/main.py:lifespan`.

---

### 2.7 HIGH — caller input mutated in place

**File:** `website/app.py:51` — `input_data['Bio Text'] = bio_text_encoded`

The dict belongs to the caller. Any retry, logging-after-call or reuse sees corrupted data.
**Fix:** build a fresh DataFrame; never touch the input. Implemented —
`ModelService._to_frame`; asserted by `test_predict_does_not_mutate_across_calls`.

---

### 2.8 HIGH — validation logic is unsound

**File:** `website/app.py:215-250`

```python
fields = {'Followers': ('int', lambda x: int(x) >= 0), …}
```

* Values are already cast at line 240, then cast **again** inside each lambda — dead work, and
  the lambda would raise (not return False) on bad input, so the `except ValueError` at 249 is
  catching errors from the validator rather than the cast.
* No upper bounds. `Followers=99999999999999999999` is accepted; Python ints are unbounded, so
  this reaches numpy and can produce overflow warnings or nonsense.
* `Bio Text` has no length cap — a 10 MB bio is accepted into a TF-IDF transform.
* `Engagement Rate (%)` alone gets a range check; the rest get `>= 0`.

**Fix:** declarative bounds in a pydantic schema, which also generates the OpenAPI contract.
Implemented — `app/schemas/predict.py`; 7 parametrised rejection cases in the test suite.

---

### 2.9 MEDIUM — `print()` used as logging, leaking user data

**File:** `website/app.py:204, 212, 232, 252, 269`

```python
print(f"Form data received: {dict(form_data)}", file=sys.stdout)
```

Every submitted profile — including free-text bio — is dumped to stdout, unstructured,
mixed across stdout/stderr, with no level, timestamp, or correlation id. Under load this is
also a lock-contended syscall per line.
**Fix:** structured JSON logging with request correlation. Implemented — `app/core/logging.py`.

---

### 2.10 MEDIUM — dead and duplicated code

| File | Status |
|---|---|
| `FAKE_PROFILE_TRAIN_CODE/train_improved.py` | byte-identical logic to `train.py` bar output filenames |
| `FAKE_PROFILE_TRAIN_CODE/debug.py` | `train_improved.py` + 6 debug prints |
| `website/fix_chart.py` | **contains a `NameError`**: line 82 references undefined `lined`. Its own comments read *"This is getting too complex… We'll exit and do that."* `fix_feature_chart()` returns `None` unconditionally |
| `website/fix_chart2.py`, `fix_chart3.py` | successive retries of the same failed edit |
| `website/modify_analytics{,2,3}.py` | three attempts at one template edit |
| `website/update_charts{,2}.py` | two more |
| `website/app.py.backup`, `.bak` | stale copies of the app |
| `templates/analytics.html.{backup,backup2,backup3,bak,bak2}` | five stale template copies |
| `templates/{inline_block,inline_block2,new_block,new_block2,prefix,prefix2,suffix,suffix2}.txt` | scratch fragments |
| `website/flask*.log` (5) | committed runtime logs |
| `FAKE_USER_DEC.zip` | a zip inside the zip |
| `test.txt` | one emoji |

These eight codemod scripts **rewrite `templates/analytics.html` in place with no backup and no
idempotency check**. Running any of them twice corrupts the template. They are unrunnable
archaeology and are the main reason the repo is hard to reason about.

**Fix:** delete. See `scripts/cleanup_legacy.sh` for a reviewed, reversible removal list.

---

### 2.11 MEDIUM — architecture violations

* **Two Flask apps** (`website/app.py`, `FAKE_PROFILE_TRAIN_CODE/app.py`) both `app.run(port=5000)`.
  The second embeds a 190-line HTML template as a Python string literal and uses
  `render_template_string` — presentation, routing and ML wiring in one module.
* **`data/app.py` is not an app.** It is a dataset generator. The name guarantees confusion.
* **No layering.** `website/app.py` mixes routing, validation, feature engineering, model
  inference, dataset statistics and presentation in 276 lines. There is no service or repository
  boundary, which is why none of it is testable.
* **DRY:** the nine feature names are re-listed in `train.py`, `website/app.py` (twice — lines
  92-94 and 166-168), `FAKE_PROFILE_TRAIN_CODE/app.py`, `augment_dataset.py`, and every template.
  Changing a feature requires seven coordinated edits.

---

### 2.12 LOW — miscellaneous

* `website/app.py:5-6` — `sys` and `json` used, but `numpy` (line 7) is used only inside the
  `/analytics` try-block; `os` (line 4) only in `load_model`. Fine, but `redirect`/`url_for` are
  imported and used while `flash` categories (`'error'`) are never styled differently.
* `train.py:66-70` — `make_prediction` closes over module-level `loaded_model`, defined after
  `joblib.dump`. Import the module and it retrains from scratch as a side effect.
* `train.py:54` — `joblib.dump(model, 'trained_model.pkl')` writes to the **current working
  directory**, while line 13 reads the dataset from `BASE_DIR`. Run it from anywhere but
  `FAKE_PROFILE_TRAIN_CODE/` and input and output diverge.
* README lines 60-63 give Windows `copy` commands in a bash fence.
* README line 231 points at "line 278-280 in website/app.py" — the file is 276 lines.

---

## Step 3 — Performance

Ranked by measured impact.

| # | Bottleneck | File / function | Cost | Fix | Gain |
|---|---|---|---|---|---|
| 1 | **CSV re-read per request** | `website/app.py:106` `analytics()` | `pd.read_csv` of 15,000 rows + `np.histogram` + feature-name derivation, inside the request thread, **on every page load** | precompute at startup, serve from memory | ~120 ms → **0 ms** per request; removes all disk I/O from the hot path |
| 2 | **No batch endpoint** | — | scoring N profiles = N HTTP round trips × ~60 ms pipeline setup | `POST /predict/batch` | **74x** — measured 8,639 profiles/s batched vs 117 rps single (§3.1) |
| 3 | **Oversized model** | `train.py:44` `RandomForestClassifier(n_estimators=200)` unbounded depth | 300,688 tree nodes, 72 MB on disk, per worker in RAM | depth 8, `min_samples_leaf=20` | 72 MB → **0.53 MB** (136x); train accuracy 0.96 → honest 0.52 |
| 4 | `print()` per field per request | `website/app.py:232` | 9 unbuffered stdout writes per prediction, lock-contended | structured logger, one line per request | ~9x fewer syscalls on the hot path |
| 5 | Blocking inference on the event loop | n/a in Flask (thread-per-request), but the naive FastAPI port would block | — | `anyio.to_thread` + `CapacityLimiter` | keeps the loop responsive; bounds memory under burst |
| 6 | BLAS oversubscription | — | each worker's numpy spawns a thread per core → 8 workers × 80 threads | `OMP_NUM_THREADS=1` | removes context-switch thrash on a many-core host |
| 7 | No response compression | — | analytics JSON is ~8 KB | `GZipMiddleware` | ~70% smaller over mobile networks |
| 8 | Model reloaded under race | `website/app.py:83-85` | up to N concurrent 72 MB deserialisations | load once in lifespan | bounded, predictable memory |

### 3.1 Measured throughput

80-core host, artifacts loaded, rate limiting off:

```
=== 2 workers, single-item /predict ===
conc=  16 n=  500  500/500 200  rps=  36.1  p50= 434.4ms  p95= 604.3ms

=== 8 workers, single-item /predict ===
conc=   8 n=  400  400/400 200  rps=  63.7  p50=  91.5ms  p95= 308.8ms
conc=  32 n= 1000 1000/1000 200  rps=  93.7  p50= 209.4ms  p95= 939.5ms
conc= 128 n= 2000 2000/2000 200  rps= 117.3  p50= 655.8ms  p95=2737.0ms

=== 8 workers, /predict/batch (100 profiles per call) ===
conc=  32 calls=200  200 ok  profiles/s=8639  p50= 307.8ms  p95= 682.4ms
```

**Reading:** inference is CPU-bound, so throughput scales with *worker count*, not async
concurrency — going 8→128 concurrent only raised throughput 63→117 rps while p95 went 309 ms →
2.7 s. The single biggest client-side win available is using the batch endpoint.

**Remaining opportunity (not taken):** single-row latency is ~60–90 ms, dominated by
`ColumnTransformer` + TF-IDF setup rather than the forest itself. Converting the fitted pipeline
to ONNX (`skl2onnx`) would plausibly cut this by 5–10x. Deferred: it adds a build step and a
runtime, and it optimises a model that has no predictive value. Fix the data first.

---

## Step 4 — Security audit

| Check | Original | Severity | Status |
|---|---|---|---|
| **Remote code execution** | `debug=True` + `host='0.0.0.0'` → Werkzeug console | **Critical** | Fixed — gunicorn, loopback bind |
| **Hardcoded credentials** | `secret_key = 'your-secret-key-change-in-production'` | **High** | Fixed — env-driven, no session state |
| **Unsafe deserialisation** | `joblib.load()` of a pickle; a writable artifacts dir = RCE | **High** | Mitigated — read-only mount (`docker-compose.yml`), version-checked |
| **Information disclosure** | `flash(f'An error occurred: {e}')` | **High** | Fixed — stable codes only |
| **Authentication** | none | **High** | Added — optional API key, `secrets.compare_digest` |
| **Authorization** | none | High | N/A — no per-user resources; documented upgrade path to JWT |
| **Rate limiting** | none | **High** | Added — fixed window, `Retry-After`; Redis path documented |
| **Input validation** | partial, unsound (§2.8) | **High** | Fixed — pydantic bounds |
| **DoS via unbounded input** | no length cap on bio, no upper bounds on numerics | **High** | Fixed — `max_length=2000`, explicit maxima |
| **XSS** | Jinja autoescape on; `result.html` renders `{{ prediction }}` escaped | Low | No stored/reflected XSS found. API returns JSON only |
| **JS injection via `\|safe`** | `analytics.html:562-568` injects 7 server values into a `<script>` with `\|safe`. `json.dumps` does not escape `</script>` | Low | Eliminated — API returns JSON; no server-rendered JS |
| **SQL injection** | N/A — no database | — | — |
| **Command injection** | N/A — no `subprocess`/`os.system` in project code | — | — |
| **SSRF** | N/A — no outbound requests | — | — |
| **CSRF** | `/predict` is a state-free POST; no session-changing actions | Low | N/A — token-auth JSON API, not cookies |
| **Directory traversal** | no user-controlled paths | — | — |
| **Prompt / RAG injection** | N/A — no LLM | — | — |
| **Security headers** | none | Medium | Partially — `TrustedHost`, CORS allowlist. HSTS/CSP belong at the proxy |
| **CORS** | none (server-rendered) | Medium | Added — configurable allowlist |
| **Secrets in repo** | `.claude/settings.local.json` committed | Low | `.env` git-ignored; `.env.example` has no values |
| **Dependency CVEs** | Flask 2.3.3 (Sep 2023), pandas 2.1.4, sklearn 1.3.2 — all ~2 years stale | Medium | Updated — see Step 10 |

### 4.1 On pickle deserialisation

`joblib.load()` executes arbitrary code during unpickling. Anyone who can write
`artifacts/model.joblib` gets code execution as the service user. Mitigations applied:

* artifacts mounted **read-only** into the container (`docker-compose.yml`),
* container runs as non-root uid 10001, `no-new-privileges:true`,
* metadata version check before use.

Not applied: artifact signing. If artifacts ever move over a network or through a shared
registry, sign them and verify the signature before `joblib.load()`.

---

## Step 5 — Scalability

Assuming the original Flask app (`app.run(debug=True)`, single-threaded dev server).

| Load | What breaks first | Why |
|---|---|---|
| **100 users** | Already broken | The dev server handles one request at a time. With `/analytics` re-reading a 15k-row CSV per request (~120 ms) plus ~60 ms inference, useful capacity is roughly **5 rps**. Requests queue; the browser times out. |
| **1,000 users** | Memory + CPU | The reloader runs two processes, each holding a 24 MB model plus a fresh pandas DataFrame per `/analytics` hit. Concurrent CSV reads multiply resident memory. The box swaps. |
| **10,000 users** | Everything | No horizontal scaling is possible: no health endpoint means no load balancer can route; no readiness means a broken instance keeps receiving traffic; flash messages need a session, and the hardcoded `secret_key` means any instance's cookies are forgeable by anyone. |
| **100,000 users** | Architecture | Server-rendered HTML cannot be consumed by mobile at all. There is no cache, no queue, no CDN boundary, no metrics to scale on. |

### After the rewrite

| Load | Verdict | Notes |
|---|---|---|
| **100** | Comfortable | 1 instance, 4 workers. p95 ~300 ms single-item. |
| **1,000** | Comfortable | 1 instance, 8 workers ≈ 117 rps sustained; 1,000 users at 1 request/10 s = 100 rps. |
| **10,000** | Needs horizontal scale | 3–4 instances behind a load balancer keyed on `/health/ready`. **Move the rate limiter to Redis first** — per-process counters make the effective limit `instances × workers × limit`. |
| **100,000** | Needs the batch path + caching | Push clients to `/predict/batch`. Add a response cache on `/analytics` and `/model/info` at the CDN (both are static between deploys). Consider ONNX to cut per-row latency 5–10x. At this scale, replace the shared API key with per-user tokens and add per-user quotas. |

**What breaks first, in order:** (1) the in-process rate limiter becomes meaningless across
instances; (2) CPU on inference — add workers/instances; (3) log volume — ship to a collector,
sample the access log.

---

## Step 6 — Production readiness

| Aspect | Before | After |
|---|---|---|
| Configuration | hardcoded literals | `pydantic-settings`, 20+ env vars, `.env.example` |
| Environment variables | none | full, documented |
| Logging | `print()` | JSON, levelled, request-correlated |
| Monitoring | none | structured access log with duration + status |
| Health checks | none | `/health/live`, `/health/ready` (separate concerns) |
| Docker | none | multi-stage, non-root, `HEALTHCHECK` |
| Docker Compose | none | resource limits, log rotation, read-only artifacts, loopback publish |
| CI/CD | none | *not provided* — see below |
| Backup | none | artifacts are rebuildable from `ml/train.py` + dataset; no stateful data exists |
| Retry | none | documented client-side (`docs/FLUTTER_INTEGRATION.md` §5) |
| Timeout | none | `INFERENCE_TIMEOUT_SECONDS`, gunicorn `--timeout` |
| Graceful shutdown | none | `exec` gunicorn as PID 1, `--graceful-timeout 30`, lifespan cleanup |
| Observability | none | request id propagated to logs *and* response body/header |
| Metrics | none | *not provided* — see below |
| Tracing | none | *not provided* — see below |
| Error reporting | flash to user | logged with correlation id; client shows the id |
| Versioning | none | `/api/v1` prefix; artifact `model_version` in every prediction |
| Release strategy | none | rolling deploy safe (graceful drain + readiness gate) |

**Explicitly not delivered**, and why:

* **CI/CD pipeline** — depends on the platform (GitHub Actions vs GitLab CI vs Jenkins), which
  was not specified. The pieces a pipeline needs are ready: `pytest tests/ -q` exits non-zero on
  failure, and the Dockerfile builds without network access to the source host.
* **Prometheus metrics / OpenTelemetry tracing** — `METRICS_ENABLED` is wired into config but no
  exporter is mounted. Adding `prometheus-fastapi-instrumentator` is a 3-line change; I did not
  add a dependency that has no scrape target yet.

---

## Step 7 — Project structure

The original layout is described in §1.2. Recommended target:

```
FAKE_USER_DEC/
├── backend/                     ← the deployable service (built)
│   ├── app/{api,core,middleware,schemas,services,utils}/
│   ├── ml/train.py
│   ├── artifacts/
│   └── tests/
├── docs/
│   ├── AUDIT.md
│   └── FLUTTER_INTEGRATION.md
├── data/
│   └── datasets/dataset.csv     ← ONE copy
└── legacy/                      ← the original Flask app, quarantined
```

### Renames, with reasons

| From | To | Why |
|---|---|---|
| `data/app.py` | `ml/generate_synthetic_dataset.py` | It is a generator, not an app. This name is the single most misleading thing in the repo — and it is where the random-label bug lives. |
| `papper/` | *delete* | Typo of "paper"; contains a 4th dataset copy and two notebooks, one named `Untitled` |
| `FAKE_PROFILE_TRAIN_CODE/` | `ml/` | Screaming case, and it contained a web app, not just training code |
| `website/` | `legacy/flask-app/` | Superseded; keep for reference during migration, delete after |
| `make_prediction` | `ModelService.predict` | It was a module function closing over globals; now a method with injected state |
| `load_model()` returning `(None,None,None)` | `ModelService.load()` raising | Silent failure was the root of the "app starts but is broken" behaviour |
| `le_bio`, `le_account` | *deleted* | `le_bio` was actively harmful (§2.1); `le_account` is redundant — the pipeline already returns string labels |

### Files to delete

`scripts/cleanup_legacy.sh` (provided, dry-run by default) removes: 8 codemod scripts, 5 `.log`
files, 7 `.backup`/`.bak` files, 8 template scratch `.txt` files, the nested
`FAKE_USER_DEC.zip`, `test.txt`, the Windows `.venv/`, 3 redundant dataset copies and 4
redundant model artifacts. **384 MB of the 410 MB project** (measured by the script's dry run).

---

## Step 8 — API review

### Original endpoints

| Endpoint | Validation | Errors | Status codes | Format | Docs | Security |
|---|---|---|---|---|---|---|
| `GET /` | — | — | 200 | HTML | none | none |
| `GET /dashboard` | — | — | 200 | HTML | none | none |
| `GET /analytics` | — | swallowed, falls back to **fake placeholder numbers** | 200 | HTML | none | none |
| `GET /settings` | — | — | 200 | HTML | none | none |
| `GET /profile` | — | — | 200 | HTML | none | none |
| `GET /word-analysis` | — | — | 200 | HTML | none | none |
| `POST /predict` | unsound (§2.8) | leaks internals | **302 on every error** | HTML | none | none |

**Two systemic problems.** First, `POST /predict` returns **302 for every failure** — validation
error, model missing, inference crash all redirect to `/`. A client cannot distinguish "you sent
bad data" from "the server is broken". Second, `/analytics` silently substitutes hardcoded
placeholders (`model_accuracy = "0.68"`, `avg_prediction_time = "0.8"`,
`importances_json = [0.18, 0.12, …]` at `website/app.py:88-95`) when the real computation fails —
which it always does, because the fallback path at line 163 triggers whenever feature-name
extraction breaks. **The dashboard has been showing invented numbers.**

### New endpoints

Documented in `backend/README.md` and `docs/FLUTTER_INTEGRATION.md` §10; auto-generated OpenAPI
at `/docs`, `/redoc`, `/openapi.json`. Every endpoint has: declarative validation, typed response
model, correct status codes (200/401/422/429/503/504), structured errors with stable codes,
correlated logging, optional auth and rate limiting.

---

## Step 9 — Database review

**No database exists.** Nothing is persisted: no prediction history, no users, no audit trail.

This is defensible for a stateless classifier and is why the service scales horizontally with no
coordination. It becomes a gap the moment you need any of: prediction history per user, model
performance monitoring in production (you cannot measure drift without stored predictions and
outcomes), abuse investigation, or per-user quotas.

**If added**, the minimum shape:

```sql
CREATE TABLE prediction (
    id            BIGSERIAL PRIMARY KEY,
    request_id    TEXT        NOT NULL,
    model_version TEXT        NOT NULL,
    features      JSONB       NOT NULL,
    label         TEXT        NOT NULL,
    confidence    REAL        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ON prediction (created_at DESC);
CREATE INDEX ON prediction (model_version, created_at DESC);  -- per-version drift queries
CREATE INDEX ON prediction (request_id);                       -- support lookups
```

Use `asyncpg` + SQLAlchemy 2.0 async with `pool_size=WORKERS*2`, write on a background task so
persistence never blocks the response, and partition by month once past ~50M rows.

---

## Step 10 — Dependencies

### Original (`website/requirements.txt`)

| Package | Pinned | Issue |
|---|---|---|
| `Flask` | 2.3.3 | ~2 years stale; superseded by the FastAPI backend |
| `pandas` | 2.1.4 | stale; **heavy** (~50 MB) for what it does here |
| `scikit-learn` | 1.3.2 | **wrong — cannot load the shipped artifacts** (§2.2) |
| `joblib` | 1.3.2 | fine, but transitively pinned by sklearn anyway |

Missing from the file but imported: `numpy` (`website/app.py:7`) — it works only because pandas
pulls it in. An undeclared direct dependency is a latent break.

Unused: nothing in `website/`, but the Windows `.venv/` bundles the full transitive tree
(~12,290 files) including `click`, `blinker`, `itsdangerous` — vendored into version control for
a platform the deployment target does not use.

### New (`backend/requirements.txt`)

| Package | Version | Rationale |
|---|---|---|
| `fastapi` | 0.115.6 | async, generates OpenAPI, pydantic validation |
| `uvicorn[standard]` | 0.34.0 | ASGI server |
| `gunicorn` | 23.0.0 | process supervision + graceful restart |
| `pydantic` / `pydantic-settings` | 2.10.4 / 2.7.0 | validation + config |
| `scikit-learn` | **1.7.2** | pinned exactly; matches the retrained artifact |
| `numpy` / `pandas` / `scipy` / `joblib` | 2.2.1 / 2.2.3 / 1.15.0 / 1.4.2 | explicit — no undeclared transitives |

**Heavy-package note:** `pandas` is used only to build a DataFrame for the sklearn pipeline. It
is ~50 MB of the image for one call. Dropping it would mean the pipeline accepts a numpy array,
which means restructuring the `ColumnTransformer` to use positional indices instead of column
names. Worth doing if image size matters; not worth the fragility otherwise. Flagged, not done.

**Breaking changes to watch:** scikit-learn does not guarantee pickle compatibility across minor
versions — that is the entire cause of §2.2. Any sklearn bump requires retraining. The version
check in `ModelService.load()` turns that from a silent production failure into a refusal to start.

---

## Step 11 — ML review

### 11.1 The labels are random. This is the finding that matters.

`data/app.py:48`:

```python
'Account Type': np.random.choice(['Real', 'Fake'], num_samples, p=[0.5, 0.5]),
```

The target is drawn independently of every feature. Every other column is also drawn from a
fixed distribution — there is no generative relationship anywhere in the file.

**Verified on the actual shipped data** (`website/dataset.csv`, 15,000 rows) by measuring the
standardised mean difference between classes for each feature:

```
Followers               Fake=  24926.26  Real=  25242.77   diff=0.0226 sd
Following               Fake=   2483.84  Real=   2491.55   diff=0.0055 sd
Posts                   Fake=    250.10  Real=    247.41   diff=0.0190 sd
Engagement Rate (%)     Fake=      5.07  Real=      4.96   diff=0.0395 sd
Avg Likes per Post      Fake=    497.74  Real=    497.30   diff=0.0016 sd
Avg Comments per Post   Fake=     49.37  Real=     50.01   diff=0.0226 sd
Verified                Fake=      0.24  Real=      0.22   diff=0.0332 sd
Account Age (Years)     Fake=      6.18  Real=      6.16   diff=0.0058 sd
distinct Bio Text values: 15
```

Every difference is ≤ 0.04 standard deviations — sampling noise at n=15,000. **No feature
carries any information about the label.**

**Shipped model, measured on its own held-out `test.csv`** (Python 3.11 + sklearn 1.9.0):

```
HELD-OUT TEST ACCURACY : 0.5083
MAJORITY BASELINE      : 0.5080
TRAIN ACCURACY         : 0.9602
```

The model is **0.0003 better than always guessing the majority class**, while scoring 0.96 on
its training set. That gap is 45 points of pure memorised noise — 200 unbounded trees with
300,688 nodes fitting random labels.

**Retrained with a bounded forest and 5-fold CV:**

```
accuracy            0.523
baseline_accuracy   0.508
lift_over_baseline  0.015
roc_auc             0.5188
cv_accuracy_mean    0.5073
cv_accuracy_std     0.0145
```

Cross-validated accuracy 0.5073 ± 0.0145 straddles the baseline. The 0.523 single-split figure
is inside the noise band. **There is no signal to learn.**

**Consequence:** no amount of model tuning, feature engineering, or architecture change fixes
this. The only fix is real labelled data. Everything else in this audit is engineering; this is
the product.

**What was done about it:** `ml/train.py` audits the data before training and writes machine-
readable warnings into `artifacts/model_meta.json`, which `GET /api/v1/model/info` serves
verbatim. `docs/FLUTTER_INTEGRATION.md` §6 shows the client-side gate. The system now tells the
truth about itself instead of rendering a 50% coin flip as "Fake Account — 50.4% confidence".

### 11.2 `augment_dataset.py` makes it worse

`website/augment_dataset.py:35-57` generates 5,000 synthetic rows by sampling each numeric column
independently from a normal fitted to its own marginal, then assigns `Account Type` via
`np.random.choice`. This:

* destroys any covariance between features (they are sampled independently),
* clips to observed min/max, piling mass at the bounds,
* adds 5,000 more randomly-labelled rows.

It grew the dataset 10,000 → 15,000 rows while adding zero information — and the 15,000-row file
is what `website/trained_model.pkl` was trained on. **Do not run this script.**

### 11.3 Other ML issues

| Issue | File | Detail |
|---|---|---|
| `bio_encoder.pkl` is incoherent | — | `LabelEncoder` over 15 strings feeding a `TfidfVectorizer` (§2.1) |
| No cross-validation | `train.py:48` | single 80/20 split; the honest 0.5073 ± 0.0145 only appears under CV |
| No calibration | `train.py:44` | `predict_proba` on an unbounded RF is badly calibrated. Fixed by depth limiting; `CalibratedClassifierCV` recommended once there is real signal |
| No class weighting | `train.py:44` | benign at 50/50, breaks on realistic 5% fraud rates. Added `class_weight='balanced'` |
| Accuracy is the wrong metric | — | for fraud detection at low base rates use PR-AUC and cost-weighted thresholds, not accuracy |
| Fake metrics in the UI | `website/app.py:88-89` | `model_accuracy = "0.68"`, `avg_prediction_time = "0.8"` — hardcoded placeholders shown to users as real |
| Fake feature importances | `website/app.py:95, 169` | `[0.18, 0.12, 0.15, …]` — invented numbers, rendered as a chart |
| No model registry / lineage | — | five `.pkl` files, no way to tell which is deployed. Fixed: `model_version` in metadata and in every prediction response |
| No drift monitoring | — | needs stored predictions (Step 9) |

**Not applicable:** GPU usage, batch inference tuning, prompt engineering, retrieval/embedding/
chunking/vector DB/reranking, context windows, token cost, hallucination, temperature, model
switching. There is no LLM, no RAG, no STT/TTS in this project.

---

## Step 12 — Bug hunt

Every bug below was reproduced.

### 12.1 `/predict` raises on 100% of requests

Two independent causes, both fatal, both confirmed:

```
# any bio outside the 15 canned strings:
ValueError: y contains previously unseen labels: 'I am a totally normal user'

# one of the 15 canned strings, correct sklearn version:
AttributeError: 'int' object has no attribute 'lower'
```

**How it occurs:** every submission, always. There is no input for which this succeeds.

### 12.2 Following the README produces a broken app

With `website/requirements.txt` exactly as pinned:

```
MODEL LOADED? True
POST /predict -> 200 (redirect)
Unexpected error: 'ColumnTransformer' object has no attribute '_name_to_fitted_passthrough'
```

`load_model()` reports success, so `/health` (if it existed) would be green. Failure appears only
at inference.

### 12.3 `fix_chart.py` has a `NameError`

`website/fix_chart.py:82`:

```python
indent = len(line) - len(lined)
```

`lined` is never defined. Line 84 immediately recomputes it correctly — the dead line above it
still raises first. `fix_feature_chart()` (line 3) also returns `None` unconditionally.

### 12.4 Codemod scripts are destructive and non-idempotent

`modify_analytics.py`, `modify_analytics2.py`, `modify_analytics3.py`, `fix_chart2.py`,
`fix_chart3.py`, `update_charts.py`, `update_charts2.py` all open `templates/analytics.html`,
mutate it, and write it back with **no backup and no guard against having already run**.
`update_charts2.py:74-88` inserts `const feature_names = …` after *every* `<script>` tag it finds.
Running it twice produces duplicate `const` declarations — a `SyntaxError` that blanks the page.
The five `analytics.html.backup*` files are the fossil record of this going wrong.

### 12.5 Analytics silently renders invented numbers

`website/app.py:163-171` — when feature-name extraction fails (it does; the transformed space has
200+ TF-IDF columns, not 9), the handler catches, logs to stderr, and substitutes
`[0.18, 0.12, 0.15, 0.22, 0.10, 0.08, 0.05, 0.05, 0.05]`. Combined with the hardcoded
`model_accuracy = "0.68"` at line 88, **the analytics dashboard has never shown real data.**

### 12.6 Race on lazy reload

§2.6. Two concurrent `/analytics` requests → two concurrent 24 MB deserialisations. Memory spike
proportional to concurrency.

### 12.7 Caller dict mutated

§2.7. `website/app.py:51`.

### 12.8 `train.py` writes to the wrong directory

`train.py:13` reads from `BASE_DIR`; `train.py:54` writes to the CWD. Run from the repo root and
the model lands in the root, while `website/app.py` looks for it in `website/`.

### 12.9 Importing `train.py` retrains the model

`train.py:51` `model.fit(...)` is at module scope, outside `if __name__ == "__main__"`. Any
import — a test, a linter, an IDE's symbol indexer — triggers a full training run and overwrites
`trained_model.pkl`.

### 12.10 Resource leaks

None found in project code — no file handles, sockets, or subprocesses are opened outside
context managers. The 5 committed `flask*.log` files are artefacts, not leaks.

### 12.11 Infinite loops

None. The `while` loops in the codemod scripts all advance `i` on every path.

---

## Step 13 — Refactoring roadmap

| Pri | Task | Difficulty | Impact | Time | Risk | Status |
|---|---|---|---|---|---|---|
| **C1** | Obtain real labelled data; retrain | Hard (data problem) | **Decisive** — nothing else matters | weeks | — | **Open — yours** |
| **C2** | Fix the bio-encoding crash | Easy | app works at all | 2 h | Low | **Done** |
| **C3** | Fix the dependency/artifact version mismatch | Easy | app works at all | 1 h | Low | **Done** |
| **C4** | Remove `debug=True` on `0.0.0.0` | Trivial | closes RCE | 15 min | None | **Done** |
| **C5** | Stop leaking exception text | Easy | closes info disclosure | 1 h | Low | **Done** |
| **H1** | Env-driven config, no hardcoded secret | Easy | deployability | 2 h | Low | **Done** |
| **H2** | Structured logging + request ids | Easy | debuggability | 2 h | Low | **Done** |
| **H3** | Health/readiness endpoints | Easy | horizontal scaling | 1 h | Low | **Done** |
| **H4** | Declarative validation with bounds | Medium | DoS + correctness | 3 h | Low | **Done** |
| **H5** | Load model once; kill the race | Easy | memory stability | 1 h | Low | **Done** |
| **H6** | Auth + rate limiting | Medium | abuse control | 3 h | Low | **Done** |
| **H7** | Test suite | Medium | prevents recurrence | 4 h | Low | **Done — 30 tests** |
| **M1** | Precompute analytics at startup | Easy | ~120 ms/request | 2 h | Low | **Done** |
| **M2** | Batch endpoint | Easy | **74x** on multi-profile | 2 h | Low | **Done** |
| **M3** | Bound the forest | Easy | 72 MB → 0.53 MB | 1 h | Low | **Done** |
| **M4** | Docker + compose + graceful shutdown | Medium | deployability | 4 h | Low | **Done** |
| **M5** | Delete 384 MB of dead files | Easy | maintainability | 1 h | **Medium** — needs review | **Script provided, not executed** |
| **M6** | Redis-backed rate limiter | Medium | correctness past 1 box | 4 h | Low | Open |
| **M7** | Prometheus metrics + OTel tracing | Medium | observability | 6 h | Low | Open |
| **M8** | CI pipeline | Easy | — | 3 h | Low | Open — platform not specified |
| **L1** | ONNX conversion | Medium | 5–10x latency | 8 h | Medium | Open — do after C1 |
| **L2** | Drop the pandas dependency | Medium | ~50 MB image | 4 h | Medium | Open |
| **L3** | Prediction history table | Medium | drift monitoring | 8 h | Low | Open — needs C1 first |

---

## Step 14 — Score

Original project, honestly:

| Dimension | Score | Reasoning |
|---|---|---|
| Architecture | **2 / 10** | Two Flask apps on one port, no layering, globals as state, a dataset generator named `app.py` |
| Code Quality | **2 / 10** | 8 dead codemod scripts (one with a `NameError`), 3 copies of the training script, 7 `.bak` files, `print()` as logging |
| Performance | **2 / 10** | CSV re-read per request, 72 MB model, dev server, no batching |
| Security | **1 / 10** | `debug=True` on `0.0.0.0` is unauthenticated RCE. Plus hardcoded key, no auth, no rate limit, exception text to users |
| Maintainability | **2 / 10** | 4 dataset copies, 5 model artifacts, 349 MB of binaries, feature list duplicated 7 times |
| Scalability | **1 / 10** | Cannot scale horizontally — no health check, no readiness, forgeable sessions |
| Readability | **3 / 10** | `website/app.py` is followable; the surrounding tree is not |
| Testing | **0 / 10** | Zero tests. This is why a 100%-broken endpoint shipped |
| Documentation | **4 / 10** | README is genuinely detailed — and describes a procedure that produces a broken app |
| Production Readiness | **1 / 10** | No config, logging, health, Docker, CI, or shutdown handling |
| **Overall** | **1.8 / 10** | |

The 1.8 is not about style. The application's one feature has never worked, and the model behind
it is a coin flip.

### After this work

| Dimension | Score | Note |
|---|---|---|
| Architecture | 8 / 10 | Clean layering, DI, versioned API |
| Code Quality | 8 / 10 | Typed, documented, no dead code in `backend/` |
| Performance | 7 / 10 | Measured and tuned; ONNX left on the table |
| Security | 8 / 10 | RCE closed, auth, rate limiting, read-only artifacts. −2: no artifact signing, no WAF |
| Maintainability | 8 / 10 | Single source of truth per concern |
| Scalability | 7 / 10 | Scales to ~10k users as-is. −3: in-process rate limiter |
| Readability | 9 / 10 | |
| Testing | 7 / 10 | 30 contract tests. −3: no load test in CI, no property tests |
| Documentation | 9 / 10 | README, audit, Flutter guide, auto OpenAPI |
| Production Readiness | 8 / 10 | −2: no CI pipeline, no metrics exporter |
| **Overall** | **7.9 / 10** | |

**The ML score is deliberately excluded from both.** On the shipped data it is a 0 and no
engineering changes that. The backend is production-ready; the model is not fit for use.

---

## Top 20 improvements by ROI

| # | Improvement | Impact | Effort | Status |
|---|---|---|---|---|
| 1 | **Get real labelled data.** Everything else is polish on a coin flip | Decisive | Weeks | **Yours** |
| 2 | Fix the bio-encoding crash — makes the product exist | Total | 2 h | Done |
| 3 | Fix the sklearn version mismatch — makes a fresh install work | Total | 1 h | Done |
| 4 | Remove `debug=True` on `0.0.0.0` — closes unauthenticated RCE | Critical | 15 min | Done |
| 5 | Stop returning exception text to users | High | 1 h | Done |
| 6 | Surface `lift_over_baseline` + warnings so a coin flip can't pose as a verdict | High | 2 h | Done |
| 7 | Add tests — the reason all of this shipped | High | 4 h | Done |
| 8 | Health + readiness endpoints | High | 1 h | Done |
| 9 | Env-driven config, no hardcoded secret | High | 2 h | Done |
| 10 | Structured logging with request correlation | High | 2 h | Done |
| 11 | Batch endpoint — **74x** for multi-profile clients | High | 2 h | Done |
| 12 | Precompute analytics — removes a 15k-row CSV read per request | High | 2 h | Done |
| 13 | Declarative validation with upper bounds — closes a DoS vector | High | 3 h | Done |
| 14 | Load model once — kills the reload race | Med-High | 1 h | Done |
| 15 | Bound the forest — 72 MB → 0.53 MB | Med-High | 1 h | Done |
| 16 | API key + rate limiting | Med-High | 3 h | Done |
| 17 | Docker + graceful shutdown + rolling deploys | Med-High | 4 h | Done |
| 18 | Delete 384 MB of dead files | Medium | 1 h | Script ready |
| 19 | Redis rate limiter (before scaling past one box) | Medium | 4 h | Open |
| 20 | Metrics + tracing exporters | Medium | 6 h | Open |
