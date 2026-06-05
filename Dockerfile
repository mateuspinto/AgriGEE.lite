# Build stage: pixi manages all deps (conda-forge + PyPI)
FROM ghcr.io/prefix-dev/pixi:latest AS build
WORKDIR /app

COPY pyproject.toml pixi.lock pixi.toml ./

# Stub agrigee_lite package so heavy pixi install layer caches even when source changes.
# pip editable install only needs the directory to exist at install time.
RUN mkdir -p agrigee_lite && touch agrigee_lite/__init__.py
RUN pixi install --frozen -e api

# Overwrite stub with real source, redo editable install (fast — deps already installed)
COPY agrigee_lite/ ./agrigee_lite/
RUN pixi run -e api python -m pip install --no-deps --no-build-isolation -e .

# Runtime stage: slim image with only the env + source
FROM debian:bookworm-slim
WORKDIR /app

COPY --from=build /app/.pixi/envs/api /app/.pixi/envs/api
COPY agrigee_lite/ ./agrigee_lite/

ENV PATH="/app/.pixi/envs/api/bin:$PATH"

# ---------------------------------------------------------------------------
# GEE credentials
# Mount the service account JSON via volume and point GEE_KEY at the path.
# Example:
#   docker run -v /host/sa.json:/secrets/sa.json:ro \
#              -e GEE_KEY=/secrets/sa.json ...
# For multiple service accounts, set GEE_KEY_MULTIPLE_ACCOUNTS to a
# comma-separated list of mounted JSON paths.
# ---------------------------------------------------------------------------
ENV GEE_KEY=""
ENV GEE_KEY_MULTIPLE_ACCOUNTS=""

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
ENV AGL_HOST="0.0.0.0"
ENV AGL_PORT="8000"

# ---------------------------------------------------------------------------
# Performance tuning — all optional, defaults match config.py
# ---------------------------------------------------------------------------
ENV AGRIGEE_MAX_PARALLEL_DOWNLOADS="40"
ENV AGRIGEE_AIMD_INITIAL_DOWNLOADS="40"
ENV AGRIGEE_AIMD_SUCCESS_STRIDE="5"
ENV AGRIGEE_EE_INTERACTIVE_CONCURRENCY="40"
ENV AGRIGEE_EE_QPS="100"
ENV AGRIGEE_EE_BATCH_CONCURRENCY="2"
ENV AGRIGEE_SITS_CHUNKSIZE="10"
ENV AGRIGEE_MAX_RETRIES_PER_CHUNK="8"
ENV AGRIGEE_MAX_URL_WORKERS="10"
ENV AGRIGEE_AIOHTTP_TIMEOUT_SECONDS="600"
ENV AGRIGEE_USE_UVLOOP="true"
ENV AGRIGEE_EE_HIGH_VOLUME_ENDPOINT="https://earthengine-highvolume.googleapis.com"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${AGL_PORT}/health')"

CMD ["sh", "-c", "agl_api --host ${AGL_HOST} --port ${AGL_PORT}"]
