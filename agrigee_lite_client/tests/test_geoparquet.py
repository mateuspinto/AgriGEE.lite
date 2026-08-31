"""Round-trips the hand-rolled GeoParquet metadata against geopandas.

geopandas is a test-only dependency (SPECS.md §9) — it's how we prove the
bytes this package builds are actually spec-compliant GeoParquet, the same
way the server (agrigee_lite/api/routes/sits.py) decodes uploads. The
package itself never imports geopandas.
"""

from __future__ import annotations

import io

import geopandas as gpd
import polars as pl
from shapely.geometry import MultiPolygon, Point, Polygon

from agrigee_lite_client._geoparquet import build_geoparquet_bytes, encode_geometries, read_plain_parquet_bytes


def test_build_geoparquet_bytes_round_trips_through_geopandas() -> None:
    geometries = [
        Point(-43.2, -22.9),
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1)])]),
    ]
    frame = pl.DataFrame(
        {
            "original_index": [0, 1, 2],
            "start_date": ["2023-01-01"] * 3,
            "end_date": ["2023-06-01"] * 3,
            "geometry": encode_geometries(geometries),
        }
    )

    content = build_geoparquet_bytes(frame)
    # geopandas' stubs only declare path: str | PathLike[str], but it accepts
    # any file-like object at runtime (it's just handed to pyarrow) — verified
    # by this very test passing.
    gdf = gpd.read_parquet(io.BytesIO(content))  # pyright: ignore[reportArgumentType]

    assert isinstance(gdf, gpd.GeoDataFrame)
    assert list(gdf["original_index"]) == [0, 1, 2]
    assert [g.geom_type for g in gdf.geometry] == ["Point", "Polygon", "MultiPolygon"]
    # Same stub gap as above cascades into this line's inferred type.
    assert gdf.geometry.iloc[0].equals(geometries[0])  # pyright: ignore[reportAttributeAccessIssue]


def test_geo_metadata_key_is_present_and_well_formed() -> None:
    import json

    import pyarrow.parquet as pq

    frame = pl.DataFrame({"geometry": encode_geometries([Point(0, 0)])})
    content = build_geoparquet_bytes(frame)
    schema = pq.read_schema(io.BytesIO(content))

    assert schema.metadata is not None
    geo_meta = json.loads(schema.metadata[b"geo"])
    assert geo_meta["primary_column"] == "geometry"
    assert geo_meta["columns"]["geometry"]["encoding"] == "WKB"


def test_read_plain_parquet_bytes_has_no_geo_dependency() -> None:
    df = pl.DataFrame({"timestamp": ["2023-01-01"], "B4": [0.1234]})
    buf = io.BytesIO()
    df.write_parquet(buf)

    result = read_plain_parquet_bytes(buf.getvalue())

    assert result.equals(df)
