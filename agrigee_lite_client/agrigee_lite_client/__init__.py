"""Thin HTTP client for the AgriGEE.lite REST API. See SPECS.md."""

from agrigee_lite_client._async_client import AsyncAgriGEEClient
from agrigee_lite_client._client import AgriGEEClient
from agrigee_lite_client._exceptions import (
    AgriGEEClientError,
    AgriGEEHTTPError,
    AgriGEEJobError,
    AgriGEEJobTimeoutError,
    AgriGEEUnknownSatelliteError,
    AgriGEEVersionMismatchError,
)
from agrigee_lite_client._version import __version__

__all__ = [
    "AgriGEEClient",
    "AgriGEEClientError",
    "AgriGEEHTTPError",
    "AgriGEEJobError",
    "AgriGEEJobTimeoutError",
    "AgriGEEUnknownSatelliteError",
    "AgriGEEVersionMismatchError",
    "AsyncAgriGEEClient",
    "__version__",
]
