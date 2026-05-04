import geopandas as gpd
import geopolars as gpl
from shapely.geometry import Point

from agrigee_lite._geo_compat import (
    geometry_to_geojson,
    get_crs,
    get_geometry_series,
    hash_geometry_row,
    iter_shapely_geometries,
    normalize_geodataframe,
    to_geojson_features,
    to_geopandas_geodataframe,
)


def _sample_geopandas() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["a", "b"], "value": [1, 2]},
        geometry=[Point(0, 0), Point(1, 1)],
        crs="EPSG:4326",
    )


def test_normalize_geodataframe_from_geopandas() -> None:
    gdf = _sample_geopandas()

    normalized = normalize_geodataframe(gdf)

    assert isinstance(normalized, gpl.GeoDataFrame)
    assert normalized.columns == ["name", "value", "geometry"]
    assert get_crs(normalized) == "EPSG:4326"


def test_normalize_geodataframe_from_geopolars() -> None:
    geopolars_gdf = normalize_geodataframe(_sample_geopandas())

    normalized = normalize_geodataframe(geopolars_gdf)

    assert isinstance(normalized, gpl.GeoDataFrame)
    assert normalized is not geopolars_gdf
    assert get_crs(normalized) == "EPSG:4326"


def test_to_geopandas_geodataframe_restores_crs() -> None:
    normalized = normalize_geodataframe(_sample_geopandas())

    restored = to_geopandas_geodataframe(normalized)

    assert isinstance(restored, gpd.GeoDataFrame)
    assert str(restored.crs) == "EPSG:4326"
    assert list(restored["name"]) == ["a", "b"]


def test_geometry_helpers_match_between_backends() -> None:
    geopandas_gdf = _sample_geopandas()
    geopolars_gdf = normalize_geodataframe(geopandas_gdf)

    geopandas_geoms = iter_shapely_geometries(geopandas_gdf)
    geopolars_geoms = iter_shapely_geometries(geopolars_gdf)

    assert [geom.wkt for geom in geopandas_geoms] == [geom.wkt for geom in geopolars_geoms]
    assert get_geometry_series(geopolars_gdf).len() == 2


def test_geojson_helpers_and_row_hash_are_backend_neutral() -> None:
    geopandas_gdf = _sample_geopandas()
    geopolars_gdf = normalize_geodataframe(geopandas_gdf)

    assert geometry_to_geojson(geopandas_gdf.geometry.iloc[0]) == {"type": "Point", "coordinates": (0.0, 0.0)}
    assert to_geojson_features(geopandas_gdf) == to_geojson_features(geopolars_gdf)
    assert hash_geometry_row(geopandas_gdf, 0) == hash_geometry_row(geopolars_gdf, 0)
