# syntax=docker/dockerfile:1.7
#
# Multi-stage: the build stage carries the compiler toolchain that numpy/scipy
# wheels occasionally need; the runtime stage carries none of it. Final image
# runs as a non-root user with a read-only-friendly layout.

# ---------- builder ---------------------------------------------------------
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
# A venv copies cleanly into the runtime stage without dragging in pip's cache.
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# ---------- runtime ---------------------------------------------------------
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Each worker already gets its own process; letting BLAS spawn a thread per
    # core inside every worker causes oversubscription and latency spikes.
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 appuser

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser ml ./ml
COPY --chown=appuser:appuser artifacts ./artifacts
COPY --chown=appuser:appuser start.sh ./
RUN chmod +x start.sh

USER appuser
EXPOSE 8000

# Probes the readiness endpoint, so an instance with a broken artifact is
# reported unhealthy instead of silently serving 503s.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://127.0.0.1:8000/health/ready || exit 1

ENTRYPOINT ["./start.sh"]
