"""Runtime configuration loaded from environment variables.

This module centralizes all tuning knobs used by AgriGEE.lite.
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        value = default
    else:
        try:
            value = int(raw)
        except ValueError:
            value = default

    if minimum is not None and value < minimum:
        value = minimum
    if maximum is not None and value > maximum:
        value = maximum
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}


# Endpoint policy: always high-volume.
HIGH_VOLUME_ENDPOINT = os.getenv("AGRIGEE_EE_HIGH_VOLUME_ENDPOINT", "https://earthengine-highvolume.googleapis.com")
EE_ENDPOINT_MODE = "highvolume"

# Quota visibility knobs.
EE_INTERACTIVE_CONCURRENCY = _env_int("AGRIGEE_EE_INTERACTIVE_CONCURRENCY", 40, minimum=1)
EE_QPS = _env_int("AGRIGEE_EE_QPS", 100, minimum=1)
EE_BATCH_CONCURRENCY = _env_int("AGRIGEE_EE_BATCH_CONCURRENCY", 2, minimum=1)

# Async runtime tuning.
USE_UVLOOP = _env_bool("AGRIGEE_USE_UVLOOP", True)
ASYNC_MAX_PARALLEL_DOWNLOADS = _env_int("AGRIGEE_MAX_PARALLEL_DOWNLOADS", 40, minimum=1)
ASYNC_MAX_RETRIES_PER_CHUNK = _env_int("AGRIGEE_MAX_RETRIES_PER_CHUNK", 5, minimum=1)
AIOHTTP_TIMEOUT_SECONDS = _env_int("AGRIGEE_AIOHTTP_TIMEOUT_SECONDS", 600, minimum=1)
AIOHTTP_CONNECTOR_LIMIT = _env_int(
    "AGRIGEE_AIOHTTP_CONNECTOR_LIMIT",
    ASYNC_MAX_PARALLEL_DOWNLOADS,
    minimum=1,
)
SITS_CHUNKSIZE = _env_int("AGRIGEE_SITS_CHUNKSIZE", 10, minimum=1)
