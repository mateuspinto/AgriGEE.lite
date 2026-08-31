"""Pure request/response helpers shared by the sync and async clients.

Nothing in this module does I/O — it only builds the bytes/dicts to send and
parses the bytes that come back. The httpx calls and the job poll loop live
in ``_client.py`` / ``_async_client.py``, since that's the only part that
actually needs to differ between sync and async (see SPECS.md §4, §8).
"""

from __future__ import annotations

import io
import json
import zipfile
from typing import Any

import polars as pl
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry

from agrigee_lite_client import _geoparquet

# ---------------------------------------------------------------------------
# sits / multiple_sits
# ---------------------------------------------------------------------------


def build_single_sits_frame(geometry: BaseGeometry, start_date: str, end_date: str) -> pl.DataFrame:
    """1-row frame used by ``get.sits()`` — see SPECS.md §6 ("naive puro")."""
    wkb = _geoparquet.encode_geometries([geometry])
    return pl.DataFrame({"start_date": [start_date], "end_date": [end_date], "geometry": wkb})


def build_multiple_sits_upload(
    frame,
    *,
    satellite: str,
    satellite_params: dict[str, Any] | None,
    reducers: list[str] | None,
    start_date_column: str,
    end_date_column: str,
    original_index_column: str,
    subsampling_max_pixels: float,
    chunksize: int,
    max_parallel_downloads: int,
    max_retries_per_chunk: int,
    force_redownload: bool,
) -> tuple[bytes, dict[str, str]]:
    """Build the multipart body for ``POST /sits/multiple/file``.

    Returns ``(parquet_bytes, form_fields)``; ``form_fields`` excludes the
    file itself, which the caller attaches separately as multipart ``file``.
    CRS is always ``EPSG:4326`` — this client never reprojects (SPECS.md §1).
    """
    parquet_bytes = _geoparquet.build_geoparquet_bytes(frame)
    form: dict[str, str] = {
        "satellite": json.dumps({"name": satellite, "params": satellite_params or {}}),
        "start_date_column": start_date_column,
        "end_date_column": end_date_column,
        "original_index_column": original_index_column,
        "subsampling_max_pixels": str(subsampling_max_pixels),
        "chunksize": str(chunksize),
        "max_parallel_downloads": str(max_parallel_downloads),
        "max_retries_per_chunk": str(max_retries_per_chunk),
        "force_redownload": str(force_redownload).lower(),
        "crs": "EPSG:4326",
    }
    if reducers is not None:
        form["reducers"] = json.dumps(reducers)
    return parquet_bytes, form


def parse_sits_download(content: bytes) -> pl.DataFrame:
    return _geoparquet.read_plain_parquet_bytes(content)


# ---------------------------------------------------------------------------
# image
# ---------------------------------------------------------------------------


def build_images_request(
    geometry: BaseGeometry,
    *,
    start_date: str,
    end_date: str,
    satellite: str,
    satellite_params: dict[str, Any] | None,
    invalid_images_threshold: float,
    max_parallel_downloads: int,
    force_redownload: bool,
    image_indices: list[int] | None,
    scale: float | None,
    dimensions: int | str | None,
) -> dict[str, Any]:
    """JSON body for ``POST /images`` — a single geometry, no parquet needed."""
    return {
        "geometry": mapping(geometry),
        "start_date": start_date,
        "end_date": end_date,
        "satellite": {"name": satellite, "params": satellite_params or {}},
        "invalid_images_threshold": invalid_images_threshold,
        "max_parallel_downloads": max_parallel_downloads,
        "force_redownload": force_redownload,
        "image_indices": image_indices,
        "scale": scale,
        "dimensions": dimensions,
    }


def parse_image_download(content: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(content))
