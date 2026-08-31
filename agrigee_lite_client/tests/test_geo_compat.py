from __future__ import annotations

import geopandas as gpd
import polars as pl
import pytest
from shapely.geometry import Point, Polygon

from agrigee_lite_client._geo_compat import to_wkb_frame
from agrigee_lite_client._geoparquet import encode_geometries

POLYGON = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_plain_polars_frame_passes_through_unchanged() -> None:
    frame = pl.DataFrame({"start_date": ["2023-01-01"], "geometry": encode_geometries([Point(0, 0)])})

    result = to_wkb_frame(frame)

    assert result is frame


def test_geopandas_geodataframe_is_converted_to_wkb() -> None:
    gdf = gpd.GeoDataFrame(
        {"start_date": ["2023-01-01", "2023-01-01"]},
        geometry=[Point(0, 0), POLYGON],
        crs="EPSG:4326",
    )

    result = to_wkb_frame(gdf)

    assert isinstance(result, pl.DataFrame)
    assert result.schema["geometry"] == pl.Binary
    assert result.get_column("start_date").to_list() == ["2023-01-01", "2023-01-01"]

    # round-trips back to the same geometries
    import shapely

    decoded = [shapely.from_wkb(b) for b in result.get_column("geometry").to_list()]
    assert decoded[0].equals(Point(0, 0))
    assert decoded[1].equals(POLYGON)


def test_geopandas_geodataframe_without_crs_is_accepted() -> None:
    gdf = gpd.GeoDataFrame({"start_date": ["2023-01-01"]}, geometry=[Point(0, 0)])  # crs=None

    result = to_wkb_frame(gdf)

    assert isinstance(result, pl.DataFrame)


def test_geopandas_geodataframe_with_non_wgs84_crs_raises() -> None:
    gdf = gpd.GeoDataFrame({"start_date": ["2023-01-01"]}, geometry=[Point(0, 0)], crs="EPSG:32723")

    with pytest.raises(ValueError, match="never reprojects"):
        to_wkb_frame(gdf)
