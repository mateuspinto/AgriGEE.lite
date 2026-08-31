"""GeoParquet encoding/decoding without geopandas. See SPECS.md §5.1.

geopolars 0.1.0-alpha.4 does not write GeoParquet file metadata on its own:
``GeoDataFrame.write_parquet`` is the plain ``polars.DataFrame.write_parquet``
(inherited, not overridden), and a geometry column round-trips through Arrow
as a bare ``Binary`` array with no CRS or extension type attached. The server
decodes uploads with ``geopandas.read_parquet``, which looks for a ``"geo"``
key in the Parquet file-level metadata
(https://geoparquet.org/releases/v1.0.0/), so this module builds that key by
hand — verified end-to-end against ``geopandas.read_parquet`` in
``tests/test_geoparquet.py``.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterable

import polars as pl
import pyarrow.parquet as pq
import shapely
from shapely.geometry.base import BaseGeometry

_GEOPARQUET_VERSION = "1.0.0"


def _geo_metadata(primary_column: str) -> bytes:
    return json.dumps(
        {
            "version": _GEOPARQUET_VERSION,
            "primary_column": primary_column,
            "columns": {
                # geometry_types=[] is valid per spec ("not specified") — this
                # client never introspects per-row geometry types.
                # crs is decorative here: the server takes the authoritative
                # CRS from the `crs` multipart form field, not this metadata
                # (see SPECS.md §5.1). Geometries must already be WGS84.
                primary_column: {"encoding": "WKB", "geometry_types": [], "crs": "OGC:CRS84"},
            },
        }
    ).encode()


def encode_geometries(geometries: Iterable[BaseGeometry]) -> list[bytes]:
    """Serialize shapely geometries to WKB bytes for a Binary parquet column."""
    return [shapely.to_wkb(geometry) for geometry in geometries]


def build_geoparquet_bytes(frame, *, geometry_column: str = "geometry") -> bytes:
    """Serialize a geopolars.GeoDataFrame or polars.DataFrame to GeoParquet bytes.

    ``geometry_column`` must already hold WKB bytes (``polars.Binary``) —
    build it with ``encode_geometries`` first.
    """
    table = frame.to_arrow()
    existing_metadata = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing_metadata, b"geo": _geo_metadata(geometry_column)})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


def read_plain_parquet_bytes(content: bytes) -> pl.DataFrame:
    """Read a plain (non-geo) parquet response, e.g. a SITS job download."""
    return pl.read_parquet(io.BytesIO(content))
