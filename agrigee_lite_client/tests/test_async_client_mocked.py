"""Async mirror of test_client_mocked.py — same FakeServer, AsyncAgriGEEClient.

Plain `asyncio.run(...)` wrappers instead of a pytest-asyncio/anyio plugin —
one less test dependency, and this package has no other async test needs.
"""

from __future__ import annotations

import asyncio

import httpx
import polars as pl
import pytest
from shapely.geometry import Point, Polygon

from agrigee_lite_client import AsyncAgriGEEClient
from agrigee_lite_client._exceptions import AgriGEEJobError, AgriGEEUnknownSatelliteError, AgriGEEVersionMismatchError
from agrigee_lite_client._geoparquet import encode_geometries

from ._fake_server import FakeServer

GEOMETRY = Point(-56.42, -11.20)
POLYGON = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def _make_client(server: FakeServer) -> AsyncAgriGEEClient:
    c = AsyncAgriGEEClient("http://test", poll_interval=0.001)
    c._http = httpx.AsyncClient(transport=httpx.MockTransport(server.handler), base_url="http://test")
    return c


def test_health() -> None:
    server = FakeServer()
    client = _make_client(server)

    async def run() -> bool:
        async with client:
            return await client.health()

    assert asyncio.run(run()) is True


def test_multiple_sits_happy_path() -> None:
    server = FakeServer()
    client = _make_client(server)
    gdf = pl.DataFrame(
        {
            "start_date": ["2023-01-01", "2023-01-01"],
            "end_date": ["2023-06-01", "2023-06-01"],
            "geometry": encode_geometries([Point(0, 0), POLYGON]),
        }
    )

    async def run() -> pl.DataFrame:
        async with client:
            return await client.get.multiple_sits(gdf, satellite="Sentinel2")

    result = asyncio.run(run())

    assert result.equals(server.sits_result)
    assert server.sits_upload_rows[-1] == 2


def test_sits_uploads_exactly_one_row() -> None:
    server = FakeServer()
    client = _make_client(server)

    async def run() -> pl.DataFrame:
        async with client:
            return await client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    result = asyncio.run(run())

    assert result.equals(server.sits_result)
    assert server.sits_upload_rows[-1] == 1


def test_image_happy_path() -> None:
    server = FakeServer()
    client = _make_client(server)

    async def run():
        async with client:
            return await client.get.image(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    zf = asyncio.run(run())

    assert zf.namelist() == ["2023-01-01.tif"]


def test_job_failure_raises_agrigee_job_error() -> None:
    server = FakeServer()
    server.queue_sits_job(statuses=["failed"], error="boom")
    client = _make_client(server)

    async def run() -> None:
        async with client:
            await client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    with pytest.raises(AgriGEEJobError) as exc_info:
        asyncio.run(run())

    assert exc_info.value.error == "boom"


def test_version_mismatch_raises_before_any_upload() -> None:
    server = FakeServer()
    server.server_version = "0.0.1-not-the-client-version"
    client = _make_client(server)

    async def run() -> None:
        async with client:
            await client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    with pytest.raises(AgriGEEVersionMismatchError):
        asyncio.run(run())

    assert server.sits_upload_rows == []


def test_unknown_satellite_raises_before_any_upload() -> None:
    server = FakeServer()
    client = _make_client(server)

    async def run() -> None:
        async with client:
            await client.get.sits(GEOMETRY, "2023-01-01", "2023-06-01", satellite="NotARealSatellite")

    with pytest.raises(AgriGEEUnknownSatelliteError):
        asyncio.run(run())

    assert server.sits_upload_rows == []


def test_dates_entirely_outside_satellite_range_raise_before_any_upload() -> None:
    server = FakeServer()
    client = _make_client(server)

    async def run() -> None:
        async with client:
            await client.get.sits(GEOMETRY, "2000-01-01", "2000-06-01", satellite="Sentinel2")

    with pytest.raises(ValueError, match="None of the requested periods intersect"):
        asyncio.run(run())

    assert server.sits_upload_rows == []


def test_multiple_sits_drops_rows_outside_satellite_range() -> None:
    server = FakeServer()
    client = _make_client(server)
    gdf = pl.DataFrame(
        {
            "start_date": ["2023-01-01", "2000-01-01"],
            "end_date": ["2023-06-01", "2000-06-01"],
            "geometry": encode_geometries([Point(0, 0), POLYGON]),
        }
    )

    async def run() -> pl.DataFrame:
        async with client:
            return await client.get.multiple_sits(gdf, satellite="Sentinel2")

    result = asyncio.run(run())

    assert result.equals(server.sits_result)
    assert server.sits_upload_rows[-1] == 1


def test_multiple_sits_accepts_geopandas_geodataframe() -> None:
    import geopandas as gpd

    server = FakeServer()
    client = _make_client(server)
    gdf = gpd.GeoDataFrame(
        {"start_date": ["2023-01-01", "2023-01-01"], "end_date": ["2023-06-01", "2023-06-01"]},
        geometry=[Point(0, 0), POLYGON],
        crs="EPSG:4326",
    )

    async def run() -> pl.DataFrame:
        async with client:
            return await client.get.multiple_sits(gdf, satellite="Sentinel2")

    result = asyncio.run(run())

    assert result.equals(server.sits_result)
    assert server.sits_upload_rows[-1] == 2
