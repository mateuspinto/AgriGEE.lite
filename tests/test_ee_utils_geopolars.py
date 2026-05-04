import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point, Polygon

from agrigee_lite._geo_compat import normalize_geodataframe
from agrigee_lite.ee_utils import _build_feature_collection_payload


def test_build_feature_collection_payload_from_geopandas() -> None:
    gdf = gpd.GeoDataFrame(
        {
            "original_index": [7, 9],
            "start_date": pd.to_datetime(["2024-01-01", "2024-02-01"]),
            "end_date": pd.to_datetime(["2024-01-10", "2024-02-10"]),
        },
        geometry=[
            Point(-46.6, -23.55),
            Polygon([(-46.61, -23.56), (-46.59, -23.56), (-46.59, -23.54), (-46.61, -23.54)]),
        ],
        crs="EPSG:4326",
    )

    payload = _build_feature_collection_payload(gdf, "original_index")

    assert payload["type"] == "FeatureCollection"
    features = payload["features"]
    assert isinstance(features, list)
    assert len(features) == 2

    point_feature = features[0]
    polygon_feature = features[1]

    assert point_feature["properties"] == {"0": 7, "s": "2024-01-01", "e": "2024-01-10"}
    assert point_feature["geometry"]["type"] == "Point"
    assert "geodesic" not in point_feature["geometry"]

    assert polygon_feature["properties"] == {"0": 9, "s": "2024-02-01", "e": "2024-02-10"}
    assert polygon_feature["geometry"]["type"] == "Polygon"
    assert polygon_feature["geometry"]["geodesic"] is True


def test_build_feature_collection_payload_from_geopolars_reprojects_to_wgs84() -> None:
    geopandas_gdf = gpd.GeoDataFrame(
        {
            "original_index": [1],
            "start_date": pd.to_datetime(["2024-03-01"]),
            "end_date": pd.to_datetime(["2024-03-05"]),
        },
        geometry=[Point(-46.6, -23.55)],
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    geopolars_gdf = normalize_geodataframe(geopandas_gdf)

    payload = _build_feature_collection_payload(geopolars_gdf, "original_index")

    features = payload["features"]
    assert isinstance(features, list)
    coords = features[0]["geometry"]["coordinates"]
    assert isinstance(coords, list)
    assert coords[0] == pytest.approx(-46.6, abs=1e-4)
    assert coords[1] == pytest.approx(-23.55, abs=1e-4)
    assert features[0]["properties"] == {"0": 1, "s": "2024-03-01", "e": "2024-03-05"}
