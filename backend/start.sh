#!/usr/bin/env bash
# Production entrypoint.
#
# gunicorn supervises N uvicorn workers: a worker that dies is replaced, and
# `--graceful-timeout` gives in-flight requests time to finish on SIGTERM, which
# is what makes a rolling deploy invisible to a mobile client.
#
# `exec` matters -- it makes gunicorn PID 1 so it receives Docker's SIGTERM
# directly instead of having it swallowed by the shell.

set -euo pipefail

HOST="${HOST:-0.0.0.0}"          # 0.0.0.0 is correct *inside* a container
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-2}"
TIMEOUT="${TIMEOUT:-60}"
GRACEFUL_TIMEOUT="${GRACEFUL_TIMEOUT:-30}"

if [[ ! -f "${ARTIFACTS_DIR:-artifacts}/${MODEL_FILENAME:-model.joblib}" ]]; then
  echo "WARNING: model artifact missing; the service will start but /health/ready will report 503." >&2
  echo "         Build one with: python -m ml.train --dataset <csv> --out artifacts/" >&2
fi

echo "starting ${WORKERS} worker(s) on ${HOST}:${PORT}"

exec gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers "${WORKERS}" \
  --bind "${HOST}:${PORT}" \
  --timeout "${TIMEOUT}" \
  --graceful-timeout "${GRACEFUL_TIMEOUT}" \
  --keep-alive 5 \
  --max-requests 2000 \
  --max-requests-jitter 200 \
  --access-logfile - \
  --error-logfile - \
  --log-level "${LOG_LEVEL:-info}"
