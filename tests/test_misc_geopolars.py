import geopandas as gpd
import geopolars as gpl
import h3
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon

from agrigee_lite._geo_compat import to_geopandas_geodataframe
from agrigee_lite.misc import create_gdf_hash, h3_clustering, simplify_gdf


def _point_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["near_b", "far", "near_a"]},
        geometry=[Point(-46.60, -23.55), Point(-43.20, -22.90), Point(-46.61, -23.56)],
        crs="EPSG:4326",
    )


def _polygon_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["poly"]},
        geometry=[Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])],
        crs="EPSG:4326",
    )


def _multipolygon_gdf() -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {"name": ["multi"]},
        geometry=[
            MultiPolygon(
                [
                    Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                    Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]),
                ]
            )
        ],
        crs="EPSG:4326",
    )


def test_simplify_gdf_leaves_point_geometry_unchanged() -> None:
    gdf = _point_gdf()

    simplified = simplify_gdf(gdf)

    assert isinstance(simplified, gpd.GeoDataFrame)
    assert list(simplified.geometry.to_wkt()) == list(gdf.geometry.to_wkt())
    assert list(simplified.index) == list(gdf.index)


def test_simplify_gdf_leaves_polygon_geometry_unchanged() -> None:
    gdf = _polygon_gdf()

    simplified = simplify_gdf(gdf)

    assert isinstance(simplified, gpd.GeoDataFrame)
    assert list(simplified.geometry.to_wkt()) == list(gdf.geometry.to_wkt())


def test_simplify_gdf_leaves_multipolygon_geometry_unchanged() -> None:
    gdf = _multipolygon_gdf()

    simplified = simplify_gdf(gdf)

    assert isinstance(simplified, gpd.GeoDataFrame)
    assert list(simplified.geometry.to_wkt()) == list(gdf.geometry.to_wkt())


def test_simplify_gdf_preserves_duplicate_geometries() -> None:
    polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    gdf = gpd.GeoDataFrame({"name": ["a", "b"]}, geometry=[polygon, polygon], crs="EPSG:4326")

    simplified = simplify_gdf(gdf)

    assert list(simplified.geometry.to_wkt()) == list(gdf.geometry.to_wkt())
    assert simplified.geometry.nunique() == 1


def test_h3_clustering_sorts_rows_and_assigns_cluster_ids() -> None:
    gdf = _point_gdf()

    clustered = h3_clustering(gdf, coarse_resolution=5, fine_resolution=8)
    clustered_gdf = to_geopandas_geodataframe(clustered).reset_index(drop=True)

    expected = []
    for row in gdf.itertuples():
        fine = h3.latlng_to_cell(row.geometry.y, row.geometry.x, 8)
        coarse = h3.cell_to_parent(fine, 5)
        expected.append((row.name, coarse, fine, row.geometry.wkt))
    expected.sort(key=lambda item: (item[1], item[2]))

    expected_cluster_ids: list[int] = []
    cluster_by_cell: dict[str, int] = {}
    for _, coarse, _, _ in expected:
        expected_cluster_ids.append(cluster_by_cell.setdefault(coarse, len(cluster_by_cell)))

    assert list(clustered_gdf["name"]) == [item[0] for item in expected]
    assert list(clustered_gdf["h3_coarse"]) == [item[1] for item in expected]
    assert list(clustered_gdf["h3_fine"]) == [item[2] for item in expected]
    assert list(clustered_gdf["cluster_id"]) == expected_cluster_ids
    assert list(clustered_gdf.geometry.to_wkt()) == [item[3] for item in expected]


def test_h3_clustering_accepts_geopolars_input() -> None:
    geopandas_gdf = _point_gdf()
    geopolars_gdf = gpl.from_geopandas(geopandas_gdf)

    clustered = h3_clustering(geopolars_gdf, coarse_resolution=5, fine_resolution=8)
    clustered_gdf = to_geopandas_geodataframe(clustered).reset_index(drop=True)

    assert isinstance(clustered, gpl.GeoDataFrame)
    assert set(clustered_gdf.columns) == {"name", "geometry", "h3_fine", "h3_coarse", "cluster_id"}
    assert not clustered_gdf.empty


def test_create_gdf_hash_matches_between_backends() -> None:
    geopandas_gdf = _point_gdf().assign(
        start_date=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        end_date=pd.to_datetime(["2024-01-10", "2024-01-11", "2024-01-12"]),
    )
    geopolars_gdf = gpl.from_geopandas(geopandas_gdf)

    geopandas_hash = create_gdf_hash(geopandas_gdf, "start_date", "end_date")
    geopolars_hash = create_gdf_hash(geopolars_gdf, "start_date", "end_date")

    assert geopandas_hash == geopolars_hash
