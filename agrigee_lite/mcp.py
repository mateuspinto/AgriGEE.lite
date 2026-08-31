"""
Optional MCP (Model Context Protocol) server for AgriGEE.lite.

Exposes the whole REST API (``agrigee_lite.api``) as MCP tools/resources, so an
LLM agent (e.g. Claude Code) can call AgriGEE.lite directly instead of shelling
out to curl. Built on top of `fastmcp <https://gofastmcp.com>`_'s FastAPI-to-MCP
converter, which walks the app's OpenAPI spec and turns every route into a tool.

Two REST behaviours don't survive a literal OpenAPI->MCP conversion, so they
get special handling:

- ``GET /jobs/{job_id}/download`` returns a raw ZIP/Parquet body. Auto-mapped
  as a *tool* it would be treated as text and its bytes corrupted, so it is
  routed to a *resource template* instead, which preserves ``response.content``
  in full.
- ``POST /sits/multiple/file`` takes a ``multipart/form-data`` upload. MCP tool
  arguments are JSON, and the OpenAPI provider doesn't base64-decode binary
  form fields, so that route is excluded and replaced by
  ``submit_sits_job_from_file``, a tool that reads the Parquet straight off
  the local disk the MCP client is already running on.

Two run modes
--------------
- **Standalone** (default — ``agl_mcp``, e.g. for ``claude mcp add``): the
  server calls the FastAPI app in-process over an ASGI transport, and owns
  Earth Engine/the cache/the job store itself, exactly like ``agl_api`` does.
- **Sidecar** (``agl_mcp --api-base-url http://host:port``, used when this
  runs alongside an already-live ``agl_api`` — e.g. the Docker image): the
  server proxies every call over real HTTP to that ``agl_api`` instance
  instead. This is required, not just a convenience — the DuckDB cache file
  only allows one process to hold a connection at a time, and job state lives
  in whichever process's in-memory ``JobStore`` created it, so a second
  independent ``init_cache()``/``job_store`` in this process would either
  crash on a lock conflict or silently see none of the jobs the API creates.

Install the extra dependencies to use this module::

    pip install agrigee_lite[mcp]
    # or
    pixi install -e mcp

Then launch the server (stdio transport, for ``claude mcp add``)::

    agl_mcp
    agl_mcp --transport http --port 8001                          # standalone, over HTTP
    agl_mcp --transport http --port 8001 --api-base-url http://127.0.0.1:8000  # sidecar
"""

from __future__ import annotations

import argparse
import asyncio
import os

try:
    import fastmcp as _fastmcp  # noqa: F401
except ImportError as exc:
    raise ImportError("agrigee_lite[mcp] is not installed. Run: pip install agrigee_lite[mcp]") from exc

import httpx
from fastmcp import FastMCP
from fastmcp.server.providers.openapi import MCPType, RouteMap

from agrigee_lite.api import create_app
from agrigee_lite.ee_utils import _install_uvloop, ee_quick_start

_ROUTE_MAPS = [
    RouteMap(methods=["POST"], pattern=r"/sits/multiple/file$", mcp_type=MCPType.EXCLUDE),
    RouteMap(methods=["GET"], pattern=r"/download$", mcp_type=MCPType.RESOURCE_TEMPLATE),
    # The HTML dashboard and its SSE stream are for humans in a browser, not MCP tool calls.
    RouteMap(methods=["GET"], pattern=r"/monitor", mcp_type=MCPType.EXCLUDE),
]


def _register_embedded_file_upload_tool(mcp: FastMCP) -> None:
    """Local-filesystem equivalent of ``POST /sits/multiple/file`` (standalone mode).

    Bypasses HTTP and the multipart encoding entirely, calling straight into the
    same job machinery the REST route uses — safe here because this process
    owns the cache and the job store outright.
    """
    import geopandas as gpd

    from agrigee_lite.api._jobs import JobStatus, JobType, job_store
    from agrigee_lite.api._models import MultipleSitsFileParams, SatelliteSpec
    from agrigee_lite.api.routes.sits import _run_multiple_sits_job_core, _sits_file_job_hash
    from agrigee_lite.config import ASYNC_MAX_PARALLEL_DOWNLOADS, ASYNC_MAX_RETRIES_PER_CHUNK, SITS_CHUNKSIZE

    @mcp.tool
    async def submit_sits_job_from_file(
        file_path: str,
        satellite_name: str = "Sentinel2",
        satellite_params: dict | None = None,
        reducers: list[str] | None = None,
        start_date_column: str = "start_date",
        end_date_column: str = "end_date",
        original_index_column: str = "original_index",
        subsampling_max_pixels: float = 1_000,
        chunksize: int = SITS_CHUNKSIZE,
        max_parallel_downloads: int = ASYNC_MAX_PARALLEL_DOWNLOADS,
        max_retries_per_chunk: int = ASYNC_MAX_RETRIES_PER_CHUNK,
        force_redownload: bool = False,
        crs: str = "EPSG:4326",
    ) -> dict:
        """
        Submit a multi-geometry SITS download job from a local Parquet file.

        Equivalent to ``POST /sits/multiple/file``, minus the multipart upload:
        reads ``file_path`` (a geopandas Parquet file with ``geometry``,
        ``start_date`` and ``end_date`` columns) directly off disk. Returns a
        ``job_id`` — poll it with ``get_job`` and fetch the result with
        ``download_job_result``.
        """
        params = MultipleSitsFileParams(
            satellite=SatelliteSpec(name=satellite_name, params=satellite_params or {}),
            reducers=reducers,
            start_date_column=start_date_column,
            end_date_column=end_date_column,
            original_index_column=original_index_column,
            subsampling_max_pixels=subsampling_max_pixels,
            chunksize=chunksize,
            max_parallel_downloads=max_parallel_downloads,
            max_retries_per_chunk=max_retries_per_chunk,
            force_redownload=force_redownload,
            crs=crs,
        )

        gdf = gpd.read_parquet(file_path)
        for col in (params.start_date_column, params.end_date_column):
            if col not in gdf.columns:
                raise ValueError(f"Column '{col}' not found in Parquet file")

        job_hash = _sits_file_job_hash(gdf, params)
        existing = job_store.get(job_hash)
        if existing is not None:
            if existing.status == JobStatus.FAILED or (
                params.force_redownload and existing.status == JobStatus.COMPLETED
            ):
                job_store.delete(job_hash)
            else:
                return {"id": existing.id, "type": existing.type, "status": existing.status}

        job = job_store.create(JobType.SITS, job_id=job_hash)
        asyncio.create_task(
            _run_multiple_sits_job_core(
                job_id=job.id,
                gdf=gdf,
                satellite_name=params.satellite.name,
                satellite_params=params.satellite.params,
                reducers=params.reducers,
                start_date_column=params.start_date_column,
                end_date_column=params.end_date_column,
                original_index_column=params.original_index_column,
                subsampling_max_pixels=params.subsampling_max_pixels,
                chunksize=params.chunksize,
                max_parallel_downloads=params.max_parallel_downloads,
                max_retries_per_chunk=params.max_retries_per_chunk,
                force_redownload=params.force_redownload,
                crs=params.crs,
            )
        )
        return {"id": job.id, "type": job.type.value, "status": job.status.value}


def _register_sidecar_file_upload_tool(mcp: FastMCP, base_url: str) -> None:
    """HTTP equivalent of ``POST /sits/multiple/file`` (sidecar mode).

    Reads the local file and uploads it as a real multipart request to the
    sibling ``agl_api`` process, so the resulting job lives where every other
    tool call (``get_job``, ``download_job_result``, ...) can see it.
    """
    import json

    from agrigee_lite.config import ASYNC_MAX_PARALLEL_DOWNLOADS, ASYNC_MAX_RETRIES_PER_CHUNK, SITS_CHUNKSIZE

    @mcp.tool
    async def submit_sits_job_from_file(
        file_path: str,
        satellite_name: str = "Sentinel2",
        satellite_params: dict | None = None,
        reducers: list[str] | None = None,
        start_date_column: str = "start_date",
        end_date_column: str = "end_date",
        original_index_column: str = "original_index",
        subsampling_max_pixels: float = 1_000,
        chunksize: int = SITS_CHUNKSIZE,
        max_parallel_downloads: int = ASYNC_MAX_PARALLEL_DOWNLOADS,
        max_retries_per_chunk: int = ASYNC_MAX_RETRIES_PER_CHUNK,
        force_redownload: bool = False,
        crs: str = "EPSG:4326",
    ) -> dict:
        """
        Submit a multi-geometry SITS download job from a local Parquet file.

        Equivalent to ``POST /sits/multiple/file``: uploads ``file_path`` (a
        geopandas Parquet file with ``geometry``, ``start_date`` and
        ``end_date`` columns) to the AgriGEE.lite API. Returns a ``job_id`` —
        poll it with ``get_job`` and fetch the result with
        ``download_job_result``.
        """
        with open(file_path, "rb") as fh:
            data = {
                "satellite": json.dumps({"name": satellite_name, "params": satellite_params or {}}),
                "start_date_column": start_date_column,
                "end_date_column": end_date_column,
                "original_index_column": original_index_column,
                "subsampling_max_pixels": str(subsampling_max_pixels),
                "chunksize": str(chunksize),
                "max_parallel_downloads": str(max_parallel_downloads),
                "max_retries_per_chunk": str(max_retries_per_chunk),
                "force_redownload": str(force_redownload).lower(),
                "crs": crs,
            }
            if reducers is not None:
                data["reducers"] = json.dumps(reducers)
            async with httpx.AsyncClient(base_url=base_url, timeout=60.0) as client:
                response = await client.post(
                    "/sits/multiple/file",
                    files={"file": (file_path, fh, "application/octet-stream")},
                    data=data,
                )
        response.raise_for_status()
        return response.json()


def create_mcp_server(base_url: str | None = None) -> FastMCP:
    """Build the MCP server.

    Pure — does not touch Earth Engine, the cache, or the job store. In
    standalone mode (``base_url=None``) that happens in ``serve()`` instead;
    in sidecar mode it never happens here at all (see module docstring).
    """
    app = create_app()

    if base_url is None:
        mcp = FastMCP.from_fastapi(app=app, name="AgriGEE.lite", route_maps=_ROUTE_MAPS)
        _register_embedded_file_upload_tool(mcp)
    else:
        client = httpx.AsyncClient(base_url=base_url, timeout=60.0)
        mcp = FastMCP.from_openapi(
            openapi_spec=app.openapi(), client=client, name="AgriGEE.lite", route_maps=_ROUTE_MAPS
        )
        _register_sidecar_file_upload_tool(mcp, base_url)

    return mcp


def serve(
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 8001,
    base_url: str | None = None,
) -> None:
    """Run the MCP server. Used as the ``agl_mcp`` CLI entry point.

    In standalone mode (``base_url=None``) this also initializes Earth Engine
    and the cache, exactly like ``agl_api`` does. In sidecar mode that's the
    sibling ``agl_api`` process's job — skipping it here is what avoids a
    DuckDB lock conflict between the two processes.
    """
    _install_uvloop()
    if base_url is None:
        from agrigee_lite.api._jobs import job_store
        from agrigee_lite.cache import init_cache

        ee_quick_start()
        init_cache()
        job_store.load_from_db()

    mcp = create_mcp_server(base_url=base_url)
    if transport == "stdio":
        mcp.run()
    else:
        mcp.run(transport=transport, host=host, port=port)


def main() -> None:
    parser = argparse.ArgumentParser(description="AgriGEE.lite MCP server")
    parser.add_argument("--transport", default="stdio", choices=["stdio", "http", "sse", "streamable-http"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--api-base-url",
        default=os.environ.get("AGL_API_BASE_URL"),
        help=(
            "URL of an already-running `agl_api` instance to proxy to (sidecar mode), "
            "e.g. http://127.0.0.1:8000. Defaults to $AGL_API_BASE_URL. "
            "Omit to run standalone, owning Earth Engine/the cache/the job store directly."
        ),
    )
    args = parser.parse_args()
    serve(transport=args.transport, host=args.host, port=args.port, base_url=args.api_base_url)
