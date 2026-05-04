import geopandas as gpd
import geopolars as gpl
import pandas as pd
from shapely.geometry import Point

from agrigee_lite.get.sits import sanitize_and_prepare_input_gdf
from agrigee_lite.sat.sentinel2 import Sentinel2


def test_sanitize_and_prepare_input_gdf_accepts_geopandas_input() -> None:
    satellite = Sentinel2()
    gdf = gpd.GeoDataFrame(
        {
            "start_date": pd.to_datetime(["2024-01-01", "2010-01-01"]),
            "end_date": pd.to_datetime(["2024-01-10", "2010-01-10"]),
        },
        geometry=[Point(-46.6, -23.55), Point(-46.7, -23.56)],
        crs="EPSG:4326",
    )

    prepared = sanitize_and_prepare_input_gdf(gdf, satellite, "original_index")

    assert isinstance(prepared, gpl.GeoDataFrame)
    assert prepared.height == 1
    assert set(prepared.columns) == {
        "geometry",
        "start_date",
        "end_date",
        "original_index",
        "h3_coarse",
        "h3_fine",
        "cluster_id",
    }
    assert prepared.get_column("original_index").to_list() == [0]


def test_sanitize_and_prepare_input_gdf_accepts_geopolars_input() -> None:
    satellite = Sentinel2()
    geopandas_gdf = gpd.GeoDataFrame(
        {
            "row_id": [101, 202],
            "start_date": pd.to_datetime(["2024-02-01", "2024-03-01"]),
            "end_date": pd.to_datetime(["2024-02-10", "2024-03-10"]),
        },
        geometry=[Point(-46.6, -23.55), Point(-43.2, -22.9)],
        crs="EPSG:4326",
    )
    geopolars_gdf = gpl.from_geopandas(geopandas_gdf)

    prepared = sanitize_and_prepare_input_gdf(geopolars_gdf, satellite, "row_id")

    assert isinstance(prepared, gpl.GeoDataFrame)
    assert prepared.height == 2
    assert prepared.get_column("row_id").to_list() == [101, 202]
    assert set(prepared.columns) == {
        "geometry",
        "start_date",
        "end_date",
        "row_id",
        "h3_coarse",
        "h3_fine",
        "cluster_id",
    }
