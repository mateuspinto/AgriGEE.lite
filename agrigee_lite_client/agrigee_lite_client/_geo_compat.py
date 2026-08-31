"""Accepts geopolars.GeoDataFrame, plain polars.DataFrame, or (if installed)
geopandas.GeoDataFrame as input to multiple_sits — geopandas is never a hard
dependency of this package (SPECS.md §3): it's detected lazily via
try/except ImportError, and only actually touched if the caller passes one.

geopolars.GeoDataFrame and plain polars.DataFrame already store their
geometry column as WKB Binary (verified against geopolars 0.1.0-alpha.4 —
see _geoparquet.py's module docstring), so both go straight to
build_geoparquet_bytes unchanged; only geopandas input needs converting.
"""

from __future__ import annotations

import polars as pl


def _is_geopandas_geodataframe(frame: object) -> bool:
    try:
        import geopandas as gpd
    except ImportError:
        return False
    return isinstance(frame, gpd.GeoDataFrame)


def _check_geopandas_crs_is_wgs84(frame) -> None:
    if frame.crs is not None and frame.crs != "EPSG:4326":
        msg = (
            f"Geometry CRS is {frame.crs!r}, but agrigee_lite_client never reprojects "
            "(SPECS.md §1) — reproject to EPSG:4326 yourself before calling multiple_sits/sits, "
            "e.g. `gdf.to_crs('EPSG:4326')`."
        )
        raise ValueError(msg)


def to_wkb_frame(frame, *, geometry_column: str = "geometry") -> pl.DataFrame:
    """Normalize any accepted frame type to a plain polars.DataFrame whose
    ``geometry_column`` holds WKB bytes, ready for build_geoparquet_bytes.
    """
    if _is_geopandas_geodataframe(frame):
        import shapely

        _check_geopandas_crs_is_wgs84(frame)
        wkb = shapely.to_wkb(frame[geometry_column].to_numpy())
        rest = pl.from_pandas(frame.drop(columns=[geometry_column]))
        return rest.with_columns(pl.Series(geometry_column, wkb))

    return frame
