from datetime import datetime

import duckdb
import geopandas as gpd
import geopolars as gpl
import pandas as pd
import polars as pl
from shapely.geometry import Point, Polygon

from agrigee_lite.cache.backend import (
    _ensure_sat_table_duck,
    _ensure_schema_duck,
    _get_band_columns,
    fetch_sits_batch_coverage,
    store_sits_polars,
)
from agrigee_lite.get.sits import sanitize_and_prepare_input_gdf
from agrigee_lite.sat.sentinel2 import Sentinel2


def _make_duckdb_conn(tmp_path) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(str(tmp_path / "cache.duckdb"))
    _ensure_schema_duck(conn)
    satellite = Sentinel2(bands={"red"})
    _ensure_sat_table_duck(conn, satellite.shortName, _get_band_columns(satellite))
    return conn


def _make_requests(geometry: Point | Polygon) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "start_date": pd.to_datetime(["2024-01-01", "2023-12-28"]),
            "end_date": pd.to_datetime(["2024-01-10", "2024-01-12"]),
        },
        geometry=[geometry, geometry],
        crs="EPSG:4326",
    )


def _assert_expected_coverage(conn: duckdb.DuckDBPyConnection, prepared_gdf: gpl.GeoDataFrame, geometry: Point | Polygon) -> None:
    satellite = Sentinel2(bands={"red"})
    job_id = store_sits_polars(
        conn,
        pl.DataFrame({"timestamp": [datetime(2024, 1, 5, 0, 0, 0)], "red": [0.25]}),
        geometry,
        "2024-01-01",
        "2024-01-10",
        satellite,
        None,
        1_000,
    )
    assert job_id is not None

    coverage = fetch_sits_batch_coverage(conn, prepared_gdf, satellite, None, 1_000, "start_date", "end_date")

    assert coverage == {
        0: ([job_id], []),
        1: ([job_id], [("2023-12-28", "2023-12-31"), ("2024-01-11", "2024-01-12")]),
    }


def test_fetch_sits_batch_coverage_accepts_geopolars_points(tmp_path) -> None:
    conn = _make_duckdb_conn(tmp_path)
    geometry = Point(-46.6, -23.55)
    satellite = Sentinel2(bands={"red"})

    geopandas_gdf = _make_requests(geometry)
    geopolars_gdf = gpl.from_geopandas(geopandas_gdf)
    prepared = sanitize_and_prepare_input_gdf(geopolars_gdf, satellite, "original_index")

    _assert_expected_coverage(conn, prepared, geometry)
    conn.close()


def test_fetch_sits_batch_coverage_accepts_geopolars_polygons(tmp_path) -> None:
    conn = _make_duckdb_conn(tmp_path)
    geometry = Polygon([(-46.61, -23.56), (-46.59, -23.56), (-46.59, -23.54), (-46.61, -23.54)])
    satellite = Sentinel2(bands={"red"})

    geopandas_gdf = _make_requests(geometry)
    geopolars_gdf = gpl.from_geopandas(geopandas_gdf)
    prepared = sanitize_and_prepare_input_gdf(geopolars_gdf, satellite, "original_index")

    _assert_expected_coverage(conn, prepared, geometry)
    conn.close()
