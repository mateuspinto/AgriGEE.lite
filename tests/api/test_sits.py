"""Integration tests for the /sits API routes.

Require live GEE credentials. Run with:
    pixi run -e api pytest -m integration tests/api/test_sits.py
"""

from __future__ import annotations

import io
import time

import geopandas as gpd
import polars as pl
import pytest
from fastapi.testclient import TestClient  # pyright: ignore[reportMissingImports]

_SAMPLE_PARQUET = "data/sample.parquet"
_SATELLITE_JSON = '{"name": "Sentinel2", "params": {}}'

# Minimal geometry from the /sits/single swagger example
_SAMPLE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [-56.421278446603054, -11.20431085146497],
            [-56.42086641797283, -11.203182131045496],
            [-56.418754238345244, -11.198938810008867],
            [-56.41853062573033, -11.198177072621217],
            [-56.41816897285581, -11.198243694391246],
            [-56.38491524890757, -11.206474250296319],
            [-56.40228720556215, -11.210026776096111],
            [-56.421278446603054, -11.20431085146497],
        ]
    ],
}

# Two-feature FeatureCollection from the /sits/multiple swagger example
_SAMPLE_FEATURE_COLLECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"start_date": "2019-10-01", "end_date": "2020-10-01"},
            "geometry": _SAMPLE_GEOMETRY,
        },
        {
            "type": "Feature",
            "properties": {"start_date": "2019-10-01", "end_date": "2020-10-01"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-56.43, -11.21],
                        [-56.42, -11.21],
                        [-56.42, -11.20],
                        [-56.43, -11.20],
                        [-56.43, -11.21],
                    ]
                ],
            },
        },
    ],
}


def _poll_job(client: TestClient, job_id: str, timeout: int = 360) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/jobs/{job_id}")
        assert r.status_code == 200
        data = r.json()
        if data["status"] in ("completed", "failed"):
            return data
        time.sleep(5)
    pytest.fail(f"Job {job_id} timed out after {timeout}s")


@pytest.mark.integration
class TestSitsRoutes:
    def test_single(self, client: TestClient) -> None:
        r = client.post(
            "/sits/single",
            json={
                "geometry": _SAMPLE_GEOMETRY,
                "start_date": "2019-10-01",
                "end_date": "2020-10-01",
                "satellite": {"name": "Sentinel2", "params": {}},
            },
        )
        assert r.status_code == 200, r.text
        data = r.json()
        assert "timestamp" in data
        assert len(data["timestamp"]) > 0

    def test_multiple_json(self, client: TestClient) -> None:
        r = client.post(
            "/sits/multiple",
            json={
                "feature_collection": _SAMPLE_FEATURE_COLLECTION,
                "satellite": {"name": "Sentinel2", "params": {}},
            },
        )
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]

        result = _poll_job(client, job_id)
        assert result["status"] == "completed", f"error: {result.get('error')}"

        dl = client.get(f"/jobs/{job_id}/download")
        assert dl.status_code == 200, dl.text
        df = pl.read_parquet(io.BytesIO(dl.content))
        assert not df.is_empty()
        assert "timestamp" in df.columns

    def test_multiple_file(self, client: TestClient) -> None:
        gdf = gpd.read_parquet(_SAMPLE_PARQUET)
        buf = io.BytesIO()
        gdf.to_parquet(buf)

        r = client.post(
            "/sits/multiple/file",
            files={"file": ("sample.parquet", buf.getvalue(), "application/octet-stream")},
            data={
                "satellite": _SATELLITE_JSON,
                "start_date_column": "start_date",
                "end_date_column": "end_date",
                "crs": "EPSG:4326",
            },
        )
        assert r.status_code == 202, r.text
        job_id = r.json()["id"]

        result = _poll_job(client, job_id)
        assert result["status"] == "completed", f"error: {result.get('error')}"

        dl = client.get(f"/jobs/{job_id}/download")
        assert dl.status_code == 200, dl.text
        df = pl.read_parquet(io.BytesIO(dl.content))
        assert not df.is_empty()
        assert "timestamp" in df.columns

    def test_multiple_file_dedup(self, client: TestClient) -> None:
        """Identical requests share the same job_id."""
        gdf = gpd.read_parquet(_SAMPLE_PARQUET)
        buf = io.BytesIO()
        gdf.to_parquet(buf)
        content = buf.getvalue()

        def _submit():
            return client.post(
                "/sits/multiple/file",
                files={"file": ("sample.parquet", content, "application/octet-stream")},
                data={"satellite": _SATELLITE_JSON, "crs": "EPSG:4326"},
            )

        r1 = _submit()
        r2 = _submit()
        assert r1.status_code == 202
        assert r2.status_code == 202
        assert r1.json()["id"] == r2.json()["id"]
