# Build stage: pixi manages all deps (conda-forge + PyPI)
# linux/amd64 only — pixi.toml platforms = ["linux-64"]
FROM --platform=linux/amd64 ghcr.io/prefix-dev/pixi:latest AS build
WORKDIR /app

COPY pyproject.toml pixi.lock pixi.toml README.md ./

# Stub agrigee_lite package so heavy pixi install layer caches even when source changes.
# pip editable install only needs the directory to exist at install time.
RUN mkdir -p agrigee_lite && touch agrigee_lite/__init__.py
RUN pixi install --frozen -e api

# Overwrite stub with real source (entry points already registered by pixi install above)
COPY agrigee_lite/ ./agrigee_lite/

# Runtime stage: slim image with only the env + source
FROM --platform=linux/amd64 debian:bookworm-slim
WORKDIR /app

COPY --from=build /app/.pixi/envs/api /app/.pixi/envs/api
COPY agrigee_lite/ ./agrigee_lite/

ENV PATH="/app/.pixi/envs/api/bin:$PATH"

# ---------------------------------------------------------------------------
# GEE credentials — pass at runtime, never bake into image:
#   docker run -v /host/sa.json:/secrets/sa.json:ro \
#              -e GEE_KEY=/secrets/sa.json ...
#   GEE_KEY_MULTIPLE_ACCOUNTS: comma-separated paths for multiple accounts
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

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${AGL_PORT}/health')"

CMD ["sh", "-c", "agl_api --host ${AGL_HOST} --port ${AGL_PORT}"]
