# syntax=docker/dockerfile:1
#
# Single-node image. Matches what the engine actually is today: one process,
# SQLite in a volume. It does not pretend to be a cluster-ready artefact --
# horizontal scaling needs the Postgres store and shared rate limiting that the
# README lists as not-done.
#
#   docker build -t adaptive-engine .
#   docker run -p 8080:8080 -v adaptive-data:/data \
#     -e ADAPTIVE_API_KEYS="dev-key:tenantA" adaptive-engine

FROM python:3.12-slim AS base

# Python-level hygiene, not preference:
#   PYTHONDONTWRITEBYTECODE -- read-only rootfs friendly
#   PYTHONUNBUFFERED        -- log_event writes one JSON line per event; buffering
#                              would hold them back and make an incident blind
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies in their own layer, pinned by requirements.txt, so a code change
# does not re-resolve numpy.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY contracts/ ./contracts/
COPY engine/ ./engine/
COPY README.md ./


# Non-root, and the data directory is the only writable path the process needs.
RUN useradd --create-home --uid 10001 engine \
    && mkdir -p /data && chown -R engine:engine /data /app
USER engine

ENV ADAPTIVE_DB=/data/adaptive.db \
    PORT=8080 \
    WORKERS=1 \
    ADAPTIVE_MAX_CONCURRENCY=2

VOLUME ["/data"]
EXPOSE 8080

# Probes /readyz, not /healthz: a container healthcheck answers "should this
# receive traffic", and that question is only answered honestly by touching the
# dependency. /healthz stays for the orchestrator's liveness probe, where a slow
# query must not get the process killed.

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import os,urllib.request,sys; \
u='http://127.0.0.1:'+os.environ.get('PORT','8080')+'/readyz'; \
sys.exit(0 if urllib.request.urlopen(u, timeout=2).status==200 else 1)"

# One worker by default, WORKERS (or `workers:` in adaptive.yaml) to raise it. A
# decide is CPU-bound Python+numpy, so in-process concurrency does not buy
# throughput: measured at 16 concurrent clients on a 20-core host, one worker
# serves 105 QPS while 8 serve 252 -- but the 8-worker tail is far worse (client
# p99 256ms against 881ms), so raising this trades predictability for volume.
# Capacity comes from workers or replicas, and each worker also has its own
# in-process rate limiter (so rate_per_sec is a per-worker share) and its own
# partial metrics view.
#
# The count is resolved by engine.config rather than expanded from the environment
# here: a mounted adaptive.yaml wins over WORKERS, and an in-memory database forces
# it to 1. Restating that precedence in shell is how the container and the app end
# up disagreeing about how the process is configured.
#
# max_concurrency (ADAPTIVE_MAX_CONCURRENCY) caps Starlette's threadpool, whose
# 40-thread default is sized for blocking I/O rather than for this. Admitting 16
# decides at once into one worker cost 2.5x throughput and 2.7x tail versus
# admitting 2 (13.4 -> 32.5 QPS, p99 1858 -> 688ms): the excess only multiplies GIL
# handoffs. 2 measured better than 4 or 8. Per worker, so leave it alone when
# raising the worker count.
CMD ["sh", "-c", "exec uvicorn engine.api:create_app --factory --host 0.0.0.0 --port ${PORT} --workers \"$(python -m engine.config workers)\" --no-access-log"]

