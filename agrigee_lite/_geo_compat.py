from __future__ import annotations

import hashlib
import json
from typing import Any, cast

import geopandas as gpd
import geopolars as gpl
import polars as pl
import pyproj
from shapely import from_wkb
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform

GeoDataFrameLike = gpd.GeoDataFrame | gpl.GeoDataFrame | pl.DataFrame
NormalizedGeoDataFrame = gpl.GeoDataFrame

_CRS_ATTR = "_agrigee_crs"


def _serialize_crs(crs: Any) -> str | None:
    if crs is None:
        return None
    return str(crs)


def _set_crs_metadata(gdf: gpl.GeoDataFrame, crs: Any) -> gpl.GeoDataFrame:
    setattr(gdf, _CRS_ATTR, _serialize_crs(crs))
    return gdf


def _coerce_geopandas_to_crs(gdf: gpd.GeoDataFrame, crs: str | None) -> gpd.GeoDataFrame:
    if gdf.crs is None and crs is None:
        raise ValueError("Input geodataframe must define a CRS or pass crs=... before conversion.")

    if crs is None:
        return gdf.copy()

    if gdf.crs is None:
        return gdf.set_crs(crs, allow_override=True)

    return gdf.to_crs(crs)


def _coerce_geopolars_to_crs(gdf: gpl.GeoDataFrame, crs: str | None) -> gpl.GeoDataFrame:
    source_crs = get_crs(gdf)
    if source_crs is None and crs is None:
        raise ValueError("Input geodataframe must define a CRS or pass crs=... before conversion.")

    target_crs = crs or source_crs
    if target_crs is None:
        raise ValueError("Input geodataframe must define a CRS or pass crs=... before conversion.")

    if crs is None or source_crs is None or str(source_crs) == crs:
        return _set_crs_metadata(cast(gpl.GeoDataFrame, gdf.clone()), target_crs)

    intermediate = gdf.to_geopandas()
    if source_crs is not None:
        intermediate = intermediate.set_crs(source_crs, allow_override=True)
    intermediate = intermediate.to_crs(crs)
    return _set_crs_metadata(cast(gpl.GeoDataFrame, gpl.from_geopandas(intermediate)), crs)


def _coerce_polars_to_geopolars(df: pl.DataFrame, *, crs: str | None, geometry_column: str) -> gpl.GeoDataFrame:
    if crs is None:
        raise ValueError("Polars input must pass crs=... before conversion to GeoPolars.")
    return _set_crs_metadata(gpl.GeoDataFrame(df, geometry=geometry_column), crs)


def get_crs(gdf: GeoDataFrameLike) -> Any | None:
    if isinstance(gdf, gpd.GeoDataFrame):
        return gdf.crs
    if isinstance(gdf, pl.DataFrame):
        return None
    return getattr(gdf, _CRS_ATTR, None)


def normalize_geodataframe(
    gdf: GeoDataFrameLike, crs: str | None = None, geometry_column: str = "geometry"
) -> gpl.GeoDataFrame:
    """
    Normalize GeoPandas or GeoPolars input to a GeoPolars GeoDataFrame.

    GeoPolars 0.1.0a4 does not preserve CRS round-trips on its own, so we
    attach the source CRS as lightweight metadata on the normalized frame.
    """
    if isinstance(gdf, gpl.GeoDataFrame):
        return _coerce_geopolars_to_crs(gdf, crs)

    if isinstance(gdf, pl.DataFrame):
        return _coerce_polars_to_geopolars(gdf, crs=crs, geometry_column=geometry_column)

    coerced = _coerce_geopandas_to_crs(gdf, crs)
    out = cast(gpl.GeoDataFrame, gpl.from_geopandas(coerced))
    return _set_crs_metadata(out, coerced.crs if crs is None else crs)


def wrap_geopolars_frame(
    df: pl.DataFrame,
    *,
    crs: Any | None = None,
    geometry_column: str = "geometry",
) -> gpl.GeoDataFrame:
    """Wrap a Polars frame back into GeoPolars and attach CRS metadata."""
    return _set_crs_metadata(gpl.GeoDataFrame(df, geometry=geometry_column), crs)


def to_geopandas_geodataframe(gdf: GeoDataFrameLike) -> gpd.GeoDataFrame:
    """Convert a compatible geo frame to GeoPandas, restoring CRS metadata."""
    if isinstance(gdf, gpd.GeoDataFrame):
        return gdf.copy()

    # Normalize first so we reliably have a GeoPolars frame to convert from.
    normalized = normalize_geodataframe(gdf)
    out = cast(gpl.GeoDataFrame, normalized).to_geopandas()
    crs = get_crs(gdf)
    if crs is not None:
        out = out.set_crs(crs, allow_override=True)
    return out


def restore_geodataframe_type(
    reference: GeoDataFrameLike,
    gdf: gpl.GeoDataFrame,
    *,
    preserve_index: bool = False,
) -> GeoDataFrameLike:
    """Return a GeoDataFrame in the same backend family as the reference."""
    if isinstance(reference, gpd.GeoDataFrame):
        out = to_geopandas_geodataframe(gdf)
        if preserve_index:
            out.index = reference.index
        return out

    if isinstance(reference, pl.DataFrame):
        return _set_crs_metadata(cast(gpl.GeoDataFrame, gpl.GeoDataFrame(gdf, geometry="geometry")), get_crs(gdf))

    return _set_crs_metadata(cast(gpl.GeoDataFrame, gdf.clone()), get_crs(gdf))


def get_geometry_series(gdf: GeoDataFrameLike) -> gpl.GeoSeries:
    """Return the geometry series from a normalized GeoPolars frame."""
    normalized = normalize_geodataframe(gdf)
    return cast(gpl.GeoSeries, normalized.geometry)


def geometry_value_to_shapely(geometry: BaseGeometry | bytes | bytearray | memoryview) -> BaseGeometry:
    """Decode a backend geometry scalar into a Shapely geometry."""
    if isinstance(geometry, BaseGeometry):
        return geometry
    return cast(BaseGeometry, from_wkb(bytes(geometry)))


def iter_shapely_geometries(gdf: GeoDataFrameLike) -> list[BaseGeometry]:
    """Materialize Shapely geometries from GeoPandas or GeoPolars input."""
    # Fast paths for concrete backends to avoid unnecessary normalization.
    if isinstance(gdf, gpd.GeoDataFrame):
        return [cast(BaseGeometry, geometry) for geometry in gdf.geometry]

    if isinstance(gdf, gpl.GeoDataFrame):
        return [geometry_value_to_shapely(geometry) for geometry in gdf.geometry.to_list()]

    # Fallback: normalize using any available CRS metadata.
    normalized = normalize_geodataframe(gdf, crs=get_crs(gdf))
    return [geometry_value_to_shapely(geometry) for geometry in cast(gpl.GeoDataFrame, normalized).geometry.to_list()]


def geometry_to_geojson(geometry: BaseGeometry | bytes | bytearray | memoryview) -> dict[str, Any]:
    """Convert a geometry scalar into a GeoJSON-ready mapping."""
    return cast(dict[str, Any], mapping(geometry_value_to_shapely(geometry)))


def transform_geometry(
    geometry: BaseGeometry,
    source_crs: str | None,
    target_crs: str = "EPSG:4326",
) -> BaseGeometry:
    """Transform a Shapely geometry between CRS definitions."""
    if source_crs is None or source_crs == target_crs:
        return geometry

    transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)
    return cast(BaseGeometry, transform(transformer.transform, geometry))


def to_geojson_features(gdf: GeoDataFrameLike, property_columns: list[str] | None = None) -> list[dict[str, Any]]:
    """Convert a compatible geo frame to GeoJSON-like feature dictionaries."""
    normalized = normalize_geodataframe(gdf)
    properties_ = property_columns or [column for column in normalized.columns if column != "geometry"]

    features: list[dict[str, Any]] = []
    for row in normalized.select(["geometry", *properties_]).iter_rows(named=True):
        geometry = row.pop("geometry")
        features.append({
            "type": "Feature",
            "geometry": geometry_to_geojson(cast(bytes | bytearray | memoryview, geometry)),
            "properties": row,
        })

    return features


def hash_geometry_row(
    gdf: GeoDataFrameLike,
    row_index: int,
    property_columns: list[str] | None = None,
) -> str:
    """Hash one geometry row in a backend-neutral, JSON-stable way."""
    normalized = normalize_geodataframe(gdf)
    properties_ = property_columns or [column for column in normalized.columns if column != "geometry"]
    row = normalized.select(["geometry", *properties_]).row(row_index, named=True)
    geometry = cast(bytes | bytearray | memoryview, row.pop("geometry"))
    payload = {
        "geometry_wkb_hex": geometry_value_to_shapely(geometry).wkb_hex,
        "properties": row,
    }
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()  # noqa: S324


__all__ = [
    "GeoDataFrameLike",
    "NormalizedGeoDataFrame",
    "geometry_to_geojson",
    "geometry_value_to_shapely",
    "get_crs",
    "get_geometry_series",
    "hash_geometry_row",
    "iter_shapely_geometries",
    "normalize_geodataframe",
    "restore_geodataframe_type",
    "transform_geometry",
    "to_geojson_features",
    "to_geopandas_geodataframe",
    "wrap_geopolars_frame",
]
