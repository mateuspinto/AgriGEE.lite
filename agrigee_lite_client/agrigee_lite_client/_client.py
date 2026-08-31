"""Synchronous AgriGEEClient. See SPECS.md §4 and §6."""

from __future__ import annotations

import contextlib
import time
import zipfile
from typing import Any

import httpx
import polars as pl
from shapely.geometry.base import BaseGeometry

from agrigee_lite_client import _geo_compat, _get, _satellite_dates, _validation
from agrigee_lite_client._exceptions import (
    AgriGEEHTTPError,
    AgriGEEJobError,
    AgriGEEJobTimeoutError,
    AgriGEEVersionMismatchError,
)
from agrigee_lite_client._version import __version__


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_error:
        detail = response.text
        with contextlib.suppress(Exception):
            detail = response.json().get("detail", detail)
        raise AgriGEEHTTPError(response.status_code, detail)


class _SyncGet:
    """Bound to a single ``AgriGEEClient`` — mirrors ``agl.get.*``."""

    def __init__(self, client: AgriGEEClient) -> None:
        self._client = client

    def sits(
        self,
        geometry: BaseGeometry,
        start_date: str,
        end_date: str,
        satellite: str,
        satellite_params: dict[str, Any] | None = None,
        reducers: list[str] | None = None,
        subsampling_max_pixels: float = 1_000,
    ) -> pl.DataFrame:
        """Naive: wraps a 1-row frame and calls ``multiple_sits`` (SPECS.md §6)."""
        frame = _get.build_single_sits_frame(geometry, start_date, end_date)
        return self.multiple_sits(
            frame,
            satellite=satellite,
            satellite_params=satellite_params,
            reducers=reducers,
            subsampling_max_pixels=subsampling_max_pixels,
        )

    def multiple_sits(
        self,
        gdf,
        satellite: str,
        satellite_params: dict[str, Any] | None = None,
        reducers: list[str] | None = None,
        start_date_column: str = "start_date",
        end_date_column: str = "end_date",
        original_index_column: str = "original_index",
        subsampling_max_pixels: float = 1_000,
        chunksize: int = 10,
        max_parallel_downloads: int = 40,
        max_retries_per_chunk: int = 8,
        force_redownload: bool = False,
    ) -> pl.DataFrame:
        self._client._ensure_server_version_matches()

        frame = _geo_compat.to_wkb_frame(gdf)
        satellite_start, satellite_end = _satellite_dates.get_satellite_date_range(satellite, satellite_params)
        validated = _validation.drop_rows_outside_satellite_range(
            frame,
            satellite_start=satellite_start,
            satellite_end=satellite_end,
            start_date_column=start_date_column,
            end_date_column=end_date_column,
        )

        parquet_bytes, form = _get.build_multiple_sits_upload(
            validated,
            satellite=satellite,
            satellite_params=satellite_params,
            reducers=reducers,
            start_date_column=start_date_column,
            end_date_column=end_date_column,
            original_index_column=original_index_column,
            subsampling_max_pixels=subsampling_max_pixels,
            chunksize=chunksize,
            max_parallel_downloads=max_parallel_downloads,
            max_retries_per_chunk=max_retries_per_chunk,
            force_redownload=force_redownload,
        )
        response = self._client._http.post(
            "/sits/multiple/file",
            files={"file": ("geometries.parquet", parquet_bytes, "application/octet-stream")},
            data=form,
        )
        _raise_for_status(response)
        job_id = response.json()["id"]
        self._client._wait_for_job(job_id)
        content = self._client._download_job(job_id)
        return _get.parse_sits_download(content)

    def image(
        self,
        geometry: BaseGeometry,
        start_date: str,
        end_date: str,
        satellite: str,
        satellite_params: dict[str, Any] | None = None,
        invalid_images_threshold: float = 0.5,
        max_parallel_downloads: int = 40,
        force_redownload: bool = False,
        image_indices: list[int] | None = None,
        scale: float | None = None,
        dimensions: int | str | None = None,
    ) -> zipfile.ZipFile:
        body = _get.build_images_request(
            geometry,
            start_date=start_date,
            end_date=end_date,
            satellite=satellite,
            satellite_params=satellite_params,
            invalid_images_threshold=invalid_images_threshold,
            max_parallel_downloads=max_parallel_downloads,
            force_redownload=force_redownload,
            image_indices=image_indices,
            scale=scale,
            dimensions=dimensions,
        )
        response = self._client._http.post("/images", json=body)
        _raise_for_status(response)
        job_id = response.json()["id"]
        self._client._wait_for_job(job_id)
        content = self._client._download_job(job_id)
        return _get.parse_image_download(content)


class AgriGEEClient:
    """Synchronous client for the AgriGEE.lite REST API. See SPECS.md."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        poll_interval: float = 1.0,
        poll_timeout: float | None = None,
    ) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout)
        self._poll_interval = poll_interval
        self._poll_timeout = poll_timeout
        self._version_checked = False
        self.get = _SyncGet(self)

    def __enter__(self) -> AgriGEEClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def health(self) -> bool:
        response = self._http.get("/health")
        _raise_for_status(response)
        return response.json().get("status") == "ok"

    def list_satellites(self) -> list[str]:
        response = self._http.get("/satellites")
        _raise_for_status(response)
        return response.json()

    def _ensure_server_version_matches(self) -> None:
        """Checked once per instance, before trusting the embedded satellite
        date table (_satellite_dates.py) — see AgriGEEVersionMismatchError.
        """
        if self._version_checked:
            return
        response = self._http.get("/version")
        _raise_for_status(response)
        server_version = response.json().get("version")
        if server_version != __version__:
            raise AgriGEEVersionMismatchError(__version__, server_version)
        self._version_checked = True

    def _wait_for_job(self, job_id: str) -> None:
        elapsed = 0.0
        while True:
            response = self._http.get(f"/jobs/{job_id}")
            _raise_for_status(response)
            job = response.json()
            status = job["status"]
            if status == "completed":
                return
            if status == "failed":
                raise AgriGEEJobError(job_id, job.get("error"))
            if self._poll_timeout is not None and elapsed >= self._poll_timeout:
                raise AgriGEEJobTimeoutError(job_id, self._poll_timeout)
            time.sleep(self._poll_interval)
            elapsed += self._poll_interval

    def _download_job(self, job_id: str) -> bytes:
        response = self._http.get(f"/jobs/{job_id}/download")
        _raise_for_status(response)
        return response.content
