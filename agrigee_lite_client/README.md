# agrigee_lite_client

Thin HTTP client for the [`agrigee_lite[api]`](../README.md) server. Doesn't
install Earth Engine, GDAL/PROJ, or DuckDB — it only knows how to build a
request, upload/download bytes, and talk to `/jobs`. See
[`SPECS.md`](SPECS.md) for the full design and the reasoning behind it.

## Install

```bash
uv pip install agrigee_lite_client
```

One fixed dependency set, always — `geopolars`, `shapely`, and `httpx`.
There's no install variation (no `[async]`, `[shapely]`, etc. extras): the
same install already covers both sync and async use.

**Use `uv`, not plain `pip`, to install this package.** `geopolars==0.1.0a4`
on PyPI ships a non-PEP-440-compliant dependency specifier
(`pyarrow>=4.0.*`), which pip 24.1+ refuses outright ("Please use pip<24.1
if you need to use this version") — `pip install agrigee_lite_client` fails
on any current pip. This is a bug in the published `geopolars` package
itself, not something this package's own metadata can work around. `uv`'s
resolver tolerates it and installs cleanly; alternatively, `pip<24.1` also
works. The same caveat applies to `agrigee_lite` (the server) itself.

## Quickstart — sync

```python
from agrigee_lite_client import AgriGEEClient
from shapely.geometry import Polygon

geometry = Polygon([(-56.42, -11.20), (-56.41, -11.20), (-56.41, -11.19)])

with AgriGEEClient("http://192.168.3.204:8100") as client:
    # a single geometry — naive: internally becomes a 1-row multiple_sits
    df = client.get.sits(geometry, "2023-01-01", "2023-06-01", satellite="Sentinel2")

    # multiple geometries — uploads a GeoParquet, polls the job, downloads the result parquet
    import polars as pl
    from agrigee_lite_client._geoparquet import encode_geometries

    gdf = pl.DataFrame({
        "start_date": ["2023-01-01", "2023-01-01"],
        "end_date": ["2023-06-01", "2023-06-01"],
        "geometry": encode_geometries([geometry, geometry]),
    })
    df = client.get.multiple_sits(gdf, satellite="Sentinel2", reducers=["mean", "std"])

    # geopandas.GeoDataFrame also works (geopandas is optional — only
    # imported if you actually pass one; it's not installed alongside the client)
    import geopandas as gpd

    gdf_geopandas = gpd.GeoDataFrame(
        {"start_date": ["2023-01-01"], "end_date": ["2023-06-01"]},
        geometry=[geometry],
        crs="EPSG:4326",
    )
    df = client.get.multiple_sits(gdf_geopandas, satellite="Sentinel2")

    # image — returns an in-memory zipfile.ZipFile, nothing decoded
    zf = client.get.image(geometry, "2023-01-01", "2023-06-01", satellite="Sentinel2")
    zf.extractall("out/")
```

## Quickstart — async

Same surface, `AsyncAgriGEEClient`:

```python
import asyncio
from agrigee_lite_client import AsyncAgriGEEClient

async def main():
    async with AsyncAgriGEEClient("http://192.168.3.204:8100") as client:
        df = await client.get.sits(geometry, "2023-01-01", "2023-06-01", satellite="Sentinel2")

asyncio.run(main())
```

## Validation before anything leaves your machine

`sits()`/`multiple_sits()` check that the requested dates actually intersect
the satellite's operating period — the same check the server does — and
drop (with a warning) rows that don't, raising `ValueError` if none are
left. All of this happens **before** building the GeoParquet or opening a
connection, using a static table embedded in the client (no network call).
An unknown satellite name is also caught here (`AgriGEEUnknownSatelliteError`).

Since that table is a copy of what the server knows, the client checks that
it's running against the same server version (via `GET /version`) before
trusting it — a mismatch raises `AgriGEEVersionMismatchError` instead of
validating against possibly stale data.

## What this client does NOT do (on purpose)

- No CRS reprojection — input geometries must be in WGS84 (EPSG:4326 /
  OGC:CRS84). If you pass a `geopandas.GeoDataFrame` with a different CRS,
  the client rejects it with a clear error instead of reprojecting.
- No local caching, no sophisticated retry/backoff — the cache already lives
  on the server; the only "retry" here is `httpx`'s transport timeout and a
  simple poll loop against `GET /jobs/{id}`.
- No image decoding — `get.image(...)` returns a raw `zipfile.ZipFile`, no
  `numpy`/`tifffile`/`rasterio`.
- `get.sits(...)` isn't a separate fast path: it's sugar over
  `get.multiple_sits(...)` with a single row, so it pays the job+poll+
  download cost even for one geometry. See `SPECS.md §6`.

## Versioning

This package's version always matches `agrigee_lite`'s (the server), same
commit — they aren't versioned independently. A test
(`tests/test_version_matches_server.py`) enforces this in CI.

## Development

```bash
cd agrigee_lite_client
uv pip install -e ".[dev]"
pytest
ruff check .
```

`tests/test_client_against_live_api.py` only runs if `AGRIGEE_TEST_API_URL`
is set — the rest (`test_client_mocked.py`, `test_async_client_mocked.py`)
use `httpx.MockTransport` and need no network.
