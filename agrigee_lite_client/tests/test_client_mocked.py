"""End-to-end tests of AgriGEEClient against a fake in-process server.

Uses httpx.MockTransport (part of httpx itself, no extra dependency) so the
whole client — multipart upload, job polling, parquet/zip parsing — runs for
real, just without a live agl_api process.
"""

from __future__ import annotations

import httpx
import polars as pl
import pytest
from shapely.geometry import Point, Polygon

from agrigee_lite_client import AgriGEEClient
from agrigee_lite_client._exceptions import (
    AgriGEEHTTPError,
    AgriGEEJobError,
    AgriGEEUnknownSatelliteError,
    AgriGEEVersionMismatchError,
)
from agrigee_lite_client._geoparquet import encode_geometries

from ._fake_server import FakeServer

GEOMETRY = Point(-56.42, -11.20)
POLYGON = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


@pytest.fixture
def server() -> FakeServer:
    return FakeServer()


@pytest.fixture
def client(server: FakeServer) -> AgriGEEClient:
    transport = httpx.MockTransport(server.handler)
    c = AgriGEEClient("http://test", poll_interval=0.001)
    c._http = httpx.Client(transport=transport, base_url="http://test")
    return c


def test_health(client: AgriGEEClient) -> None:
    assert client.health() is True


def test_list_satellites(client: AgriGEEClient) -> None:
    assert client.list_satellites() == ["Landsat8", "Sentinel2"]


def test_multiple_sits_happy_path(client: AgriGEEClient, server: FakeServer) -> None:
    gdf = pl.DataFrame(
        {
            "start_date": ["2023-01-01", "2023-01-01"],
            "end_date": ["2023-06-01", "2023-06-01"],
            "geometry": encode_geometries([Point(0, 0), POLYGON]),
        }
    )

    result = client.get.multiple_sits(gdf, satellite="Sentinel2")

    assert result.equals(server.sits_result)
    assert server.sits_upload_rows[-1] == 2


def test_sits_uploads_exactly_one_row(client: AgriGEEClient, server: FakeServer) -> None:
    """The naive sits() wrapper always goes through multiple_sits with 1 row (SPECS.md §6)."""
    result = client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    assert result.equals(server.sits_result)
    assert server.sits_upload_rows[-1] == 1


def test_image_happy_path(client: AgriGEEClient, server: FakeServer) -> None:
    zf = client.get.image(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2", satellite_params={"a": 1})

    assert zf.namelist() == ["2023-01-01.tif"]
    assert zf.read("2023-01-01.tif") == b"fake-tiff-bytes"
    assert server.image_requests[-1]["satellite"] == {"name": "Sentinel2", "params": {"a": 1}}
    assert server.image_requests[-1]["geometry"]["type"] == "Point"


def test_multiple_sits_polls_until_completed(client: AgriGEEClient, server: FakeServer) -> None:
    server.queue_sits_job(statuses=["pending", "running", "running", "completed"])

    result = client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    assert result.equals(server.sits_result)


def test_job_failure_raises_agrigee_job_error(client: AgriGEEClient, server: FakeServer) -> None:
    server.queue_sits_job(statuses=["failed"], error="Earth Engine quota exceeded")

    with pytest.raises(AgriGEEJobError) as exc_info:
        client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    assert exc_info.value.error == "Earth Engine quota exceeded"


def test_version_mismatch_raises_before_any_upload(client: AgriGEEClient, server: FakeServer) -> None:
    server.server_version = "0.0.1-not-the-client-version"

    with pytest.raises(AgriGEEVersionMismatchError) as exc_info:
        client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    assert exc_info.value.server_version == "0.0.1-not-the-client-version"
    assert server.sits_upload_rows == []


def test_unknown_satellite_raises_before_any_upload(client: AgriGEEClient, server: FakeServer) -> None:
    with pytest.raises(AgriGEEUnknownSatelliteError):
        client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="NotARealSatellite")

    assert server.sits_upload_rows == []


def test_dates_entirely_outside_satellite_range_raise_before_any_upload(
    client: AgriGEEClient, server: FakeServer
) -> None:
    """Sentinel2 starts 2019-01-01 — a 2000 request should never reach the network."""
    with pytest.raises(ValueError, match="None of the requested periods intersect"):
        client.get.sits(GEOMETRY, "2000-01-01", "2000-06-01", satellite="Sentinel2")

    assert server.sits_upload_rows == []


def test_multiple_sits_drops_rows_outside_satellite_range(client: AgriGEEClient, server: FakeServer) -> None:
    gdf = pl.DataFrame(
        {
            "start_date": ["2023-01-01", "2000-01-01"],  # second row predates Sentinel2 entirely
            "end_date": ["2023-06-01", "2000-06-01"],
            "geometry": encode_geometries([Point(0, 0), POLYGON]),
        }
    )

    result = client.get.multiple_sits(gdf, satellite="Sentinel2")

    assert result.equals(server.sits_result)
    assert server.sits_upload_rows[-1] == 1  # only the in-range row was uploaded


def test_multiple_sits_accepts_geopandas_geodataframe(client: AgriGEEClient, server: FakeServer) -> None:
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {
            "start_date": ["2023-01-01", "2023-01-01"],
            "end_date": ["2023-06-01", "2023-06-01"],
        },
        geometry=[Point(0, 0), POLYGON],
        crs="EPSG:4326",
    )

    result = client.get.multiple_sits(gdf, satellite="Sentinel2")

    assert result.equals(server.sits_result)
    assert server.sits_upload_rows[-1] == 2


def test_multiple_sits_rejects_non_wgs84_geopandas_crs(client: AgriGEEClient, server: FakeServer) -> None:
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"start_date": ["2023-01-01"], "end_date": ["2023-06-01"]},
        geometry=[Point(0, 0)],
        crs="EPSG:32723",
    )

    with pytest.raises(ValueError, match="never reprojects"):
        client.get.multiple_sits(gdf, satellite="Sentinel2")

    assert server.sits_upload_rows == []


def test_http_error_raises_agrigee_http_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "Column 'start_date' not found in Parquet file"})

    client = AgriGEEClient("http://test")
    client._http = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")

    with pytest.raises(AgriGEEHTTPError) as exc_info:
        client.list_satellites()

    assert exc_info.value.status_code == 422
    assert "start_date" in exc_info.value.detail
