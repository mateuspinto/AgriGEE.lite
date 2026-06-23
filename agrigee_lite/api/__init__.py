"""
Optional FastAPI server for AgriGEE.lite.

Install the extra dependencies to use this module::

    pip install agrigee_lite[api]
    # or
    pixi install -e api

Then launch the server::

    agl_api                          # defaults: host=127.0.0.1, port=8000
    agl_api --host 0.0.0.0 --port 8080
    # or programmatically:
    import agrigee_lite.api as agl_api
    agl_api.serve()

Architecture
------------
All download endpoints are **non-blocking**:

- ``POST /images`` and ``POST /sits/multiple`` accept the request, create a job,
  launch an ``asyncio.Task``, and return **202 Accepted** with a ``job_id``
  immediately — even if the download takes hours.
- ``GET /jobs/{job_id}`` lets callers poll status (pending → running → completed/failed)
  and retrieve the result once finished.
- ``POST /sits/single`` is synchronous-style (runs in a thread pool) because a
  single-geometry request is fast enough to block for.

Scalability notes
-----------------
The in-memory ``JobStore`` is suitable for a single-process deployment.
For horizontal scaling, replace it with a Redis-backed store (same interface)
and run multiple uvicorn workers or deploy behind a load balancer.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

try:
    import fastapi as _fastapi  # noqa: F401
    import uvicorn as _uvicorn  # noqa: F401
except ImportError as exc:
    raise ImportError("agrigee_lite[api] is not installed. " "Run: pip install agrigee_lite[api]") from exc

from fastapi import FastAPI

from agrigee_lite.api._satellites import REGISTRY
from agrigee_lite.api.routes import router
from agrigee_lite.ee_utils import _install_uvloop, ee_quick_start


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    from agrigee_lite.api._jobs import job_store
    from agrigee_lite.cache import init_cache
    ee_quick_start()
    init_cache()
    job_store.load_from_db()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="AgriGEE.lite API",
        description=(
            "REST API for downloading satellite imagery and time series via Google Earth Engine. "
            "Long-running jobs return 202 Accepted; poll `/jobs/{job_id}` for status."
        ),
        version="1.0.0",
        lifespan=_lifespan,
    )

    @app.get("/satellites", tags=["meta"])
    async def list_satellites() -> list[str]:
        """List all available satellite names accepted by the download endpoints."""
        return sorted(REGISTRY)

    @app.get("/health", tags=["meta"])
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(router)
    return app


def serve(host: str = "127.0.0.1", port: int = 8000, reload: bool = False) -> None:
    """Launch the uvicorn server. Used as the ``agl_api`` CLI entry point."""
    import uvicorn

    _install_uvloop()
    uvicorn.run(
        "agrigee_lite.api:create_app",
        factory=True,
        host=host,
        port=port,
        reload=reload,
    )


def main() -> None:
    """CLI entry point — parses --host, --port, --reload."""
    import argparse

    parser = argparse.ArgumentParser(description="AgriGEE.lite API server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    serve(host=args.host, port=args.port, reload=args.reload)
