from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import pathlib
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast

import duckdb
import geopandas as gpd
import h3
import pandas as pd
import polars as pl
import sqlalchemy as sa
from sqlalchemy.pool import NullPool

from agrigee_lite.sat.abstract_satellite import AbstractSatellite

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = pathlib.Path.home() / ".cache" / "agrigee_lite" / "sits_cache.duckdb"
_PG_DB_NAME = "agrigeelite"

_duck_conn: duckdb.DuckDBPyConnection | None = None
_pg_engine: sa.Engine | None = None

CacheEngine = duckdb.DuckDBPyConnection | sa.Engine

_DUCK_SYSTEM = {"geometries", "sits_jobs"}
_PG_SYSTEM = {"geometries", "sits_jobs", "spatial_ref_sys", "geometry_columns"}


# ---------------------------------------------------------------------------
# Engine accessor
# ---------------------------------------------------------------------------


def get_engine() -> CacheEngine | None:
    if _duck_conn is not None:
        return _duck_conn
    return _pg_engine


def _pg_env_set() -> bool:
    return all(k in os.environ for k in ("AGRIGEE_PG_HOST", "AGRIGEE_PG_USER", "AGRIGEE_PG_PASSWORD"))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _compute_geom_hash(geometry: Any) -> str:
    return hashlib.sha1(geometry.wkb).hexdigest()  # noqa: S324


compute_geom_hash = _compute_geom_hash


def _compute_job_hash(geom_hash: str, params_hash: str, start_date: str, end_date: str) -> str:
    return hashlib.sha1(f"{geom_hash}|{params_hash}|{start_date}|{end_date}".encode()).hexdigest()  # noqa: S324


def _compute_params_hash(
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> str:
    obj = {
        "satellite": satellite.log_dict(),
        "reducers": sorted(reducers) if reducers else None,
        "subsampling_max_pixels": subsampling_max_pixels,
    }
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()  # noqa: S324


def _geom_type_str(geometry: Any) -> str:
    t = geometry.geom_type
    if t == "Point":
        return "point"
    if t == "Polygon":
        return "polygon"
    if t in ("MultiPolygon", "MultiLineString", "MultiPoint"):
        return "multipolygon"
    return "geometry"


def _repr_point(geometry: Any) -> tuple[float, float]:
    if geometry.geom_type == "Point":
        return float(geometry.x), float(geometry.y)
    c = geometry.centroid
    return float(c.x), float(c.y)


def _compute_h3_for_point(x: float, y: float) -> tuple[str, str]:
    fine = h3.latlng_to_cell(y, x, 8)
    coarse = h3.cell_to_parent(fine, 5)
    return coarse, fine


def _compute_gaps(
    query_start: str,
    query_end: str,
    covered: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    qs = date.fromisoformat(query_start)
    qe = date.fromisoformat(query_end)
    if not covered:
        return [(query_start, query_end)]
    intervals = sorted((date.fromisoformat(s), date.fromisoformat(e)) for s, e in covered)
    merged: list[tuple[date, date]] = []
    for s, e in intervals:
        s = max(s, qs)
        e = min(e, qe)
        if s > e:
            continue
        if merged and s <= merged[-1][1] + timedelta(days=1):
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    gaps: list[tuple[str, str]] = []
    current = qs
    for s, e in merged:
        if s > current:
            gaps.append((current.isoformat(), (s - timedelta(days=1)).isoformat()))
        current = e + timedelta(days=1)
    if current <= qe:
        gaps.append((current.isoformat(), qe.isoformat()))
    return gaps


def _get_band_columns(satellite: AbstractSatellite) -> list[str]:
    if satellite.availableBands:
        return list(satellite.availableBands.keys())
    if satellite.toDownloadSelectors:
        return [s.split("_", 1)[1] if "_" in s else s for s in satellite.toDownloadSelectors]
    if hasattr(satellite, "classes"):
        return [f"soil_{info['label'].lower()}" for info in satellite.classes.values()]  # type: ignore[attr-defined]
    return []


def _chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _normalize_timestamp_pl(df: pl.DataFrame) -> pl.DataFrame:
    if "timestamp" not in df.columns:
        return df
    return df.with_columns(pl.col("timestamp").cast(pl.Datetime, strict=False))


# ---------------------------------------------------------------------------
# DuckDB — schema
# ---------------------------------------------------------------------------


def _make_duckdb_conn(db_path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _ensure_schema_duck(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SEQUENCE IF NOT EXISTS geometries_id_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geometries (
            id           BIGINT  PRIMARY KEY DEFAULT nextval('geometries_id_seq'),
            geom_hash    TEXT    NOT NULL UNIQUE,
            geometry     BLOB,
            repr_point_x DOUBLE  NOT NULL DEFAULT 0,
            repr_point_y DOUBLE  NOT NULL DEFAULT 0,
            geom_type    TEXT    NOT NULL DEFAULT 'geometry',
            h3_coarse    TEXT    NOT NULL DEFAULT '',
            h3_fine      TEXT    NOT NULL DEFAULT ''
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_geom_hash  ON geometries(geom_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_geom_h3c   ON geometries(h3_coarse)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_geom_h3f   ON geometries(h3_fine)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_geom_point ON geometries(repr_point_x, repr_point_y)")

    conn.execute("CREATE SEQUENCE IF NOT EXISTS sits_jobs_id_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sits_jobs (
            id                     BIGINT PRIMARY KEY DEFAULT nextval('sits_jobs_id_seq'),
            job_hash               TEXT   NOT NULL UNIQUE,
            geometry_id            BIGINT NOT NULL REFERENCES geometries(id),
            satellite_short_name   TEXT   NOT NULL,
            params_hash            TEXT   NOT NULL,
            reducers               TEXT,
            subsampling_max_pixels DOUBLE NOT NULL,
            start_date             TEXT   NOT NULL,
            end_date               TEXT   NOT NULL,
            fetched_at             TIMESTAMPTZ NOT NULL,
            UNIQUE (geometry_id, satellite_short_name, params_hash, start_date, end_date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_hash ON sits_jobs(job_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_geom ON sits_jobs(geometry_id, satellite_short_name, params_hash)")


def _ensure_sat_table_duck(conn: duckdb.DuckDBPyConnection, table_name: str, band_cols: list[str]) -> None:
    if not band_cols:
        return
    seq_name = f"{table_name}_id_seq"
    conn.execute(f'CREATE SEQUENCE IF NOT EXISTS "{seq_name}"')
    band_defs = ",\n    ".join(f'"{c}" DOUBLE' for c in band_cols)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id        BIGINT PRIMARY KEY DEFAULT nextval('"{seq_name}"'),
            job_id    BIGINT NOT NULL REFERENCES sits_jobs(id),
            timestamp TIMESTAMPTZ,
            {band_defs}
        )
    """)
    conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_jid" ON "{table_name}"(job_id)')


# ---------------------------------------------------------------------------
# DuckDB — reads
# ---------------------------------------------------------------------------


def _fetch_sits_with_gaps_duck(
    conn: duckdb.DuckDBPyConnection,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> tuple[pl.DataFrame, list[tuple[str, str]]]:
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    band_cols = _get_band_columns(satellite)
    table_name = satellite.shortName

    if geometry.geom_type == "Point":
        rx, ry = _repr_point(geometry)
        geom_row = conn.execute(
            "SELECT id FROM geometries WHERE repr_point_x = ? AND repr_point_y = ?", [rx, ry]
        ).fetchone()
    else:
        geom_hash = _compute_geom_hash(geometry)
        geom_row = conn.execute("SELECT id FROM geometries WHERE geom_hash = ?", [geom_hash]).fetchone()

    if geom_row is None:
        return pl.DataFrame(), [(start_date, end_date)]

    geom_id = int(geom_row[0])

    job_rows = conn.execute(
        """
        SELECT id, start_date, end_date FROM sits_jobs
        WHERE geometry_id = ? AND satellite_short_name = ? AND params_hash = ?
          AND start_date <= ? AND end_date >= ?
        """,
        [geom_id, satellite.shortName, params_hash, end_date, start_date],
    ).fetchall()

    if not job_rows:
        return pl.DataFrame(), [(start_date, end_date)]

    covered = [(str(r[1]), str(r[2])) for r in job_rows]
    job_ids = [int(r[0]) for r in job_rows]
    gaps = _compute_gaps(start_date, end_date, covered)

    cols_sql = ", ".join(f'"{c}"' for c in band_cols)
    ph = ", ".join("?" * len(job_ids))
    pl_df = conn.execute(
        f'SELECT timestamp, {cols_sql} FROM "{table_name}" '
        f"WHERE job_id IN ({ph}) AND timestamp >= ? AND timestamp <= ? "
        f"ORDER BY timestamp",
        [*job_ids, start_date, end_date],
    ).pl()

    if pl_df.is_empty():
        return pl.DataFrame(), gaps

    return _normalize_timestamp_pl(pl_df), gaps


def _fetch_sits_by_jids_duck(
    conn: duckdb.DuckDBPyConnection,
    satellite: AbstractSatellite,
    job_ids: list[int],
) -> dict[int, pl.DataFrame]:
    if not job_ids:
        return {}

    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)
    cols_sql = ", ".join(f'"{c}"' for c in band_cols)
    result: dict[int, pl.DataFrame] = {}

    for chunk in _chunked(list(dict.fromkeys(job_ids)), 400):
        ph = ", ".join("?" * len(chunk))
        pl_df = conn.execute(
            f'SELECT job_id, timestamp, {cols_sql} FROM "{table_name}"'
            f" WHERE job_id IN ({ph}) ORDER BY job_id, timestamp",
            chunk,
        ).pl()

        for sub in pl_df.partition_by("job_id", maintain_order=True):
            jid = int(sub["job_id"][0])
            result[jid] = _normalize_timestamp_pl(sub.drop("job_id"))

    return result


def _fetch_sits_batch_coverage_duck(
    conn: duckdb.DuckDBPyConnection,
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    start_date_col: str,
    end_date_col: str,
) -> dict[int, tuple[list[int], list[tuple[str, str]]]]:
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)

    h3_coarse_vals = list(gdf["h3_coarse"].astype(str).unique())
    h3_fine_vals = list(gdf["h3_fine"].astype(str).unique())
    if not h3_coarse_vals:
        return {}

    c_ph = ", ".join("?" * len(h3_coarse_vals))
    f_ph = ", ".join("?" * len(h3_fine_vals))
    candidate_rows = conn.execute(
        f"""
        SELECT id, geom_hash, h3_fine, geom_type, repr_point_x, repr_point_y
        FROM geometries
        WHERE h3_coarse IN ({c_ph}) AND h3_fine IN ({f_ph})
        """,
        [*h3_coarse_vals, *h3_fine_vals],
    ).fetchall()

    if not candidate_rows:
        return {}

    h3_fine_to_geoms: dict[str, dict[str, int]] = {}
    point_to_geom: dict[tuple[float, float], int] = {}
    for crow in candidate_rows:
        geom_id, geom_hash, h3_fine, geom_type, rx, ry = crow
        h3_fine_to_geoms.setdefault(str(h3_fine), {})[str(geom_hash)] = int(geom_id)
        if geom_type == "point":
            point_to_geom[(float(rx), float(ry))] = int(geom_id)

    candidate_h3_fines = set(h3_fine_to_geoms.keys())
    gdf_h3_fine = gdf["h3_fine"].astype(str)
    mask_h3 = gdf_h3_fine.isin(candidate_h3_fines)
    if not mask_h3.any():
        return {}

    is_point = gdf.geometry.geom_type == "Point"
    hash_mask = mask_h3 & ~is_point
    point_mask = mask_h3 & is_point

    geom_hash_series: pd.Series[Any] = pd.Series(index=gdf.index, dtype=object)
    if hash_mask.any():
        geom_hash_series[hash_mask] = gdf.loc[hash_mask, "geometry"].apply(_compute_geom_hash)

    geom_id_series: pd.Series[Any] = pd.Series(index=gdf.index, dtype=object)

    for pos in gdf.index[hash_mask]:
        gh = geom_hash_series.at[pos]
        h3f = str(gdf_h3_fine.at[pos])
        gid = h3_fine_to_geoms.get(h3f, {}).get(str(gh))
        if gid is not None:
            geom_id_series.at[pos] = gid

    for pos in gdf.index[point_mask]:
        geom: Any = gdf.at[pos, "geometry"]
        gid = point_to_geom.get((float(geom.x), float(geom.y)))
        if gid is not None:
            geom_id_series.at[pos] = gid

    valid_mask = geom_id_series.notna()
    if not valid_mask.any():
        return {}

    unique_geom_ids = [int(gid) for gid in geom_id_series[valid_mask].astype(int).unique().tolist()]
    gid_ph = ", ".join("?" * len(unique_geom_ids))
    job_rows = conn.execute(
        f"""
        SELECT geometry_id, id, start_date, end_date
        FROM sits_jobs
        WHERE geometry_id IN ({gid_ph})
          AND satellite_short_name = ? AND params_hash = ?
        """,
        [*unique_geom_ids, satellite.shortName, params_hash],
    ).fetchall()

    jobs_by_geom: dict[int, list[tuple[int, str, str]]] = {}
    for jrow in job_rows:
        gid_ = int(jrow[0])
        jobs_by_geom.setdefault(gid_, []).append((int(jrow[1]), str(jrow[2]), str(jrow[3])))

    result: dict[int, tuple[list[int], list[tuple[str, str]]]] = {}
    for pos in gdf.index[valid_mask]:
        gid = int(cast(int, geom_id_series.at[pos]))
        geom_jobs = jobs_by_geom.get(gid)
        if not geom_jobs:
            continue
        q_start = str(gdf.at[pos, start_date_col])[:10]
        q_end = str(gdf.at[pos, end_date_col])[:10]
        overlapping = [(jid, js, je) for jid, js, je in geom_jobs if js <= q_end and je >= q_start]
        if not overlapping:
            continue
        job_ids = [jid for jid, _, _ in overlapping]
        covered = [(js, je) for _, js, je in overlapping]
        gaps = _compute_gaps(q_start, q_end, covered)
        result[int(pos)] = (job_ids, gaps)

    return result


# ---------------------------------------------------------------------------
# DuckDB — write
# ---------------------------------------------------------------------------


def _store_sits_duck(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> int | None:
    if df.empty:
        return None

    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    job_hash = _compute_job_hash(geom_hash, params_hash, start_date, end_date)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    rx, ry = _repr_point(geometry)
    gtype = _geom_type_str(geometry)
    h3_coarse, h3_fine = _compute_h3_for_point(rx, ry)

    conn.begin()
    try:
        conn.execute(
            """
            INSERT INTO geometries (geom_hash, geometry, repr_point_x, repr_point_y, geom_type, h3_coarse, h3_fine)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (geom_hash) DO NOTHING
            """,
            [geom_hash, geometry.wkb, rx, ry, gtype, h3_coarse, h3_fine],
        )
        geom_id_row = conn.execute("SELECT id FROM geometries WHERE geom_hash = ?", [geom_hash]).fetchone()
        assert geom_id_row is not None
        geom_id: int = int(geom_id_row[0])

        conn.execute(
            """
            INSERT INTO sits_jobs
              (job_hash, geometry_id, satellite_short_name, params_hash, reducers,
               subsampling_max_pixels, start_date, end_date, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (geometry_id, satellite_short_name, params_hash, start_date, end_date) DO NOTHING
            """,
            [
                job_hash, geom_id, satellite.shortName, params_hash,
                json.dumps(sorted(reducers)) if reducers else None,
                subsampling_max_pixels, start_date, end_date,
                datetime.now(UTC).isoformat(),
            ],
        )
        job_id_row = conn.execute(
            """SELECT id FROM sits_jobs WHERE geometry_id = ? AND satellite_short_name = ?
               AND params_hash = ? AND start_date = ? AND end_date = ?""",
            [geom_id, satellite.shortName, params_hash, start_date, end_date],
        ).fetchone()
        assert job_id_row is not None
        job_id: int = int(job_id_row[0])

        already = conn.execute(
            f'SELECT 1 FROM "{table_name}" WHERE job_id = ? LIMIT 1', [job_id]
        ).fetchone()
        if already:
            conn.commit()
            return job_id

        present_band_cols = [c for c in band_cols if c in df.columns]
        ts_col = "timestamp" if "timestamp" in df.columns else None
        col_names = ", ".join(["job_id", "timestamp", *[f'"{c}"' for c in present_band_cols]])
        ph = ", ".join(["?"] * (2 + len(present_band_cols)))

        rows: list[list[Any]] = []
        for raw in df.to_dict("records"):
            ts_val = str(raw[ts_col]) if ts_col and pd.notna(raw[ts_col]) else None
            rows.append([job_id, ts_val, *[None if pd.isna(raw[c]) else float(raw[c]) for c in present_band_cols]])

        if rows:
            conn.executemany(f'INSERT INTO "{table_name}" ({col_names}) VALUES ({ph})', rows)

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return job_id


def _store_sits_duck_polars(
    conn: duckdb.DuckDBPyConnection,
    df: pl.DataFrame,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> int | None:
    if df.is_empty():
        return None

    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    job_hash = _compute_job_hash(geom_hash, params_hash, start_date, end_date)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    rx, ry = _repr_point(geometry)
    gtype = _geom_type_str(geometry)
    h3_coarse, h3_fine = _compute_h3_for_point(rx, ry)

    conn.begin()
    try:
        conn.execute(
            """
            INSERT INTO geometries (geom_hash, geometry, repr_point_x, repr_point_y, geom_type, h3_coarse, h3_fine)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (geom_hash) DO NOTHING
            """,
            [geom_hash, geometry.wkb, rx, ry, gtype, h3_coarse, h3_fine],
        )
        geom_id_row = conn.execute("SELECT id FROM geometries WHERE geom_hash = ?", [geom_hash]).fetchone()
        assert geom_id_row is not None
        geom_id: int = int(geom_id_row[0])

        conn.execute(
            """
            INSERT INTO sits_jobs
              (job_hash, geometry_id, satellite_short_name, params_hash, reducers,
               subsampling_max_pixels, start_date, end_date, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (geometry_id, satellite_short_name, params_hash, start_date, end_date) DO NOTHING
            """,
            [
                job_hash, geom_id, satellite.shortName, params_hash,
                json.dumps(sorted(reducers)) if reducers else None,
                subsampling_max_pixels, start_date, end_date,
                datetime.now(UTC).isoformat(),
            ],
        )
        job_id_row = conn.execute(
            """SELECT id FROM sits_jobs WHERE geometry_id = ? AND satellite_short_name = ?
               AND params_hash = ? AND start_date = ? AND end_date = ?""",
            [geom_id, satellite.shortName, params_hash, start_date, end_date],
        ).fetchone()
        assert job_id_row is not None
        job_id: int = int(job_id_row[0])

        already = conn.execute(
            f'SELECT 1 FROM "{table_name}" WHERE job_id = ? LIMIT 1', [job_id]
        ).fetchone()
        if already:
            conn.commit()
            return job_id

        present_band_cols = [c for c in band_cols if c in df.columns]
        obs_pl = (
            df.select(["timestamp", *present_band_cols])
            .with_columns(pl.lit(job_id).alias("job_id"))
            .cast({"timestamp": pl.Utf8})
        )
        col_order = ["job_id", "timestamp", *present_band_cols]
        obs_pl = obs_pl.select(col_order)
        col_sql = ", ".join(f'"{column}"' for column in col_order)

        conn.register("_obs_tmp", obs_pl.to_arrow())
        try:
            conn.execute(
                f'INSERT INTO "{table_name}" ({col_sql}) '
                f"SELECT {col_sql} FROM _obs_tmp"
            )
        finally:
            conn.unregister("_obs_tmp")

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return job_id


# ---------------------------------------------------------------------------
# PostGIS — schema
# ---------------------------------------------------------------------------


def _make_pg_engine() -> sa.Engine:
    host = os.environ["AGRIGEE_PG_HOST"]
    user = os.environ["AGRIGEE_PG_USER"]
    password = os.environ["AGRIGEE_PG_PASSWORD"]
    port = os.environ.get("AGRIGEE_PG_PORT", "5432")

    def _url(db: str) -> str:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    admin = sa.create_engine(_url("postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool)
    with admin.connect() as conn:
        exists = conn.execute(
            sa.text("SELECT 1 FROM pg_database WHERE datname = :db"),
            {"db": _PG_DB_NAME},
        ).fetchone()
        if not exists:
            conn.execute(sa.text(f'CREATE DATABASE "{_PG_DB_NAME}"'))
            logger.info("PostGIS cache: created database %r", _PG_DB_NAME)
    admin.dispose()
    return sa.create_engine(_url(_PG_DB_NAME), poolclass=NullPool)


def _ensure_postgis_extension(conn: sa.Connection) -> None:
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS postgis"))


def _ensure_geometries_table_pg(conn: sa.Connection) -> None:
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS geometries (
            id           SERIAL PRIMARY KEY,
            geom_hash    TEXT   NOT NULL UNIQUE,
            geometry     geometry(Geometry, 4326),
            repr_point_x DOUBLE PRECISION NOT NULL DEFAULT 0,
            repr_point_y DOUBLE PRECISION NOT NULL DEFAULT 0,
            geom_type    TEXT   NOT NULL DEFAULT 'geometry',
            h3_coarse    TEXT   NOT NULL DEFAULT '',
            h3_fine      TEXT   NOT NULL DEFAULT ''
        )
    """))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_geom_hash  ON geometries (geom_hash)"))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_geom_h3c   ON geometries (h3_coarse)"))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_geom_h3f   ON geometries (h3_fine)"))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_geom_point ON geometries (repr_point_x, repr_point_y)"))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_geom_gist  ON geometries USING GIST (geometry)"))


def _ensure_sits_jobs_table_pg(conn: sa.Connection) -> None:
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS sits_jobs (
            id                     SERIAL PRIMARY KEY,
            job_hash               TEXT   NOT NULL UNIQUE,
            geometry_id            INTEGER NOT NULL REFERENCES geometries(id),
            satellite_short_name   TEXT   NOT NULL,
            params_hash            TEXT   NOT NULL,
            reducers               TEXT,
            subsampling_max_pixels REAL   NOT NULL,
            start_date             TEXT   NOT NULL,
            end_date               TEXT   NOT NULL,
            fetched_at             TIMESTAMPTZ NOT NULL,
            UNIQUE (geometry_id, satellite_short_name, params_hash, start_date, end_date)
        )
    """))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_jobs_hash ON sits_jobs (job_hash)"))
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_jobs_geom "
        "ON sits_jobs (geometry_id, satellite_short_name, params_hash)"
    ))


def _ensure_satellite_table_pg(conn: sa.Connection, table_name: str, band_columns: list[str]) -> None:
    if not band_columns:
        return
    band_cols_sql = ",\n    ".join(f'"{col}" REAL' for col in band_columns)
    conn.execute(sa.text(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id        SERIAL PRIMARY KEY,
            job_id    INTEGER NOT NULL REFERENCES sits_jobs(id),
            timestamp TIMESTAMPTZ,
            {band_cols_sql}
        )
    """))
    conn.execute(sa.text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_jid" ON "{table_name}" (job_id)'))
    conn.execute(sa.text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_ts"  ON "{table_name}" (timestamp)'))


# ---------------------------------------------------------------------------
# PostGIS — reads
# ---------------------------------------------------------------------------


def _fetch_sits_with_gaps_pg(
    engine: sa.Engine,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> tuple[pl.DataFrame, list[tuple[str, str]]]:
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    band_cols = _get_band_columns(satellite)
    table_name = satellite.shortName

    with engine.connect() as conn:
        if geometry.geom_type == "Point":
            rx, ry = _repr_point(geometry)
            geom_row = conn.execute(
                sa.text("SELECT id FROM geometries WHERE repr_point_x = :rx AND repr_point_y = :ry"),
                {"rx": rx, "ry": ry},
            ).fetchone()
        else:
            geom_hash = _compute_geom_hash(geometry)
            geom_row = conn.execute(
                sa.text("SELECT id FROM geometries WHERE geom_hash = :h"), {"h": geom_hash}
            ).fetchone()

        if geom_row is None:
            return pl.DataFrame(), [(start_date, end_date)]

        geom_id = int(geom_row[0])

        job_rows = conn.execute(
            sa.text("""
                SELECT id, start_date, end_date FROM sits_jobs
                WHERE geometry_id = :gid AND satellite_short_name = :sat AND params_hash = :ph
                  AND start_date <= :ed AND end_date >= :sd
            """),
            {"gid": geom_id, "sat": satellite.shortName, "ph": params_hash, "sd": start_date, "ed": end_date},
        ).fetchall()

        if not job_rows:
            return pl.DataFrame(), [(start_date, end_date)]

        covered = [(str(r[1]), str(r[2])) for r in job_rows]
        job_ids = [int(r[0]) for r in job_rows]
        gaps = _compute_gaps(start_date, end_date, covered)

        cols_sql = ", ".join(f'"{c}"' for c in band_cols)
        placeholders = ", ".join(f":id{i}" for i in range(len(job_ids)))
        params: dict[str, Any] = {f"id{i}": jid for i, jid in enumerate(job_ids)}
        params["sd"] = start_date
        params["ed"] = end_date
        rows = conn.execute(
            sa.text(
                f'SELECT timestamp, {cols_sql} FROM "{table_name}" '
                f"WHERE job_id IN ({placeholders}) "
                f"AND timestamp >= :sd AND timestamp <= :ed "
                f"ORDER BY timestamp"
            ),
            params,
        ).fetchall()

    if not rows:
        return pl.DataFrame(), gaps

    df = pl.DataFrame(rows, schema=["timestamp", *band_cols], orient="row")
    return _normalize_timestamp_pl(df), gaps


def _fetch_sits_by_jids_pg(
    engine: sa.Engine,
    satellite: AbstractSatellite,
    job_ids: list[int],
) -> dict[int, pl.DataFrame]:
    if not job_ids:
        return {}

    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)
    cols_sql = ", ".join(f'"{c}"' for c in band_cols)
    result: dict[int, pl.DataFrame] = {}
    unique_job_ids = list(dict.fromkeys(job_ids))

    with engine.connect() as conn:
        for chunk in _chunked(unique_job_ids, 400):
            placeholders = ", ".join(f":id{i}" for i in range(len(chunk)))
            params: dict[str, int] = {f"id{i}": jid for i, jid in enumerate(chunk)}
            rows = conn.execute(
                sa.text(
                    f'SELECT job_id, timestamp, {cols_sql} FROM "{table_name}"'
                    f" WHERE job_id IN ({placeholders})"
                    f" ORDER BY job_id, timestamp"
                ),
                params,
            ).fetchall()

            current_id: int | None = None
            current_rows: list[Any] = []
            for row in rows:
                jid = int(row[0])
                if jid != current_id:
                    if current_id is not None and current_rows:
                        df = pl.DataFrame(current_rows, schema=["timestamp", *band_cols], orient="row")
                        result[current_id] = _normalize_timestamp_pl(df)
                    current_id = jid
                    current_rows = []
                current_rows.append(row[1:])
            if current_id is not None and current_rows:
                df = pl.DataFrame(current_rows, schema=["timestamp", *band_cols], orient="row")
                result[current_id] = _normalize_timestamp_pl(df)

    return result


def _fetch_sits_batch_coverage_pg(
    engine: sa.Engine,
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    start_date_col: str,
    end_date_col: str,
) -> dict[int, tuple[list[int], list[tuple[str, str]]]]:
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)

    h3_coarse_vals = list(gdf["h3_coarse"].astype(str).unique())
    h3_fine_vals = list(gdf["h3_fine"].astype(str).unique())
    if not h3_coarse_vals:
        return {}

    with engine.connect() as conn:
        candidate_rows = conn.execute(
            sa.text("""
                SELECT id, geom_hash, h3_fine, geom_type, repr_point_x, repr_point_y
                FROM geometries
                WHERE h3_coarse = ANY(:coarse) AND h3_fine = ANY(:fine)
            """),
            {"coarse": h3_coarse_vals, "fine": h3_fine_vals},
        ).fetchall()

        if not candidate_rows:
            return {}

        h3_fine_to_geoms: dict[str, dict[str, int]] = {}
        point_to_geom: dict[tuple[float, float], int] = {}
        for crow in candidate_rows:
            geom_id, geom_hash, h3_fine, geom_type, rx, ry = crow
            h3_fine_to_geoms.setdefault(str(h3_fine), {})[str(geom_hash)] = int(geom_id)
            if geom_type == "point":
                point_to_geom[(float(rx), float(ry))] = int(geom_id)

        candidate_h3_fines = set(h3_fine_to_geoms.keys())
        gdf_h3_fine = gdf["h3_fine"].astype(str)
        mask_h3 = gdf_h3_fine.isin(candidate_h3_fines)
        if not mask_h3.any():
            return {}

        is_point = gdf.geometry.geom_type == "Point"
        hash_mask = mask_h3 & ~is_point
        point_mask = mask_h3 & is_point

        geom_hash_series: pd.Series[Any] = pd.Series(index=gdf.index, dtype=object)
        if hash_mask.any():
            geom_hash_series[hash_mask] = gdf.loc[hash_mask, "geometry"].apply(_compute_geom_hash)

        geom_id_series: pd.Series[Any] = pd.Series(index=gdf.index, dtype=object)

        for pos in gdf.index[hash_mask]:
            gh = geom_hash_series.at[pos]
            h3f = str(gdf_h3_fine.at[pos])
            gid = h3_fine_to_geoms.get(h3f, {}).get(str(gh))
            if gid is not None:
                geom_id_series.at[pos] = gid

        for pos in gdf.index[point_mask]:
            geom: Any = gdf.at[pos, "geometry"]
            gid = point_to_geom.get((float(geom.x), float(geom.y)))
            if gid is not None:
                geom_id_series.at[pos] = gid

        valid_mask = geom_id_series.notna()
        if not valid_mask.any():
            return {}

        unique_geom_ids = [int(gid) for gid in geom_id_series[valid_mask].astype(int).unique().tolist()]
        job_rows = conn.execute(
            sa.text("""
                SELECT geometry_id, id, start_date, end_date FROM sits_jobs
                WHERE geometry_id = ANY(:gids)
                  AND satellite_short_name = :sat AND params_hash = :ph
            """),
            {"gids": unique_geom_ids, "sat": satellite.shortName, "ph": params_hash},
        ).fetchall()

    jobs_by_geom: dict[int, list[tuple[int, str, str]]] = {}
    for jrow in job_rows:
        gid_ = int(jrow[0])
        jobs_by_geom.setdefault(gid_, []).append((int(jrow[1]), str(jrow[2]), str(jrow[3])))

    result: dict[int, tuple[list[int], list[tuple[str, str]]]] = {}
    for pos in gdf.index[valid_mask]:
        gid = int(cast(int, geom_id_series.at[pos]))
        geom_jobs = jobs_by_geom.get(gid)
        if not geom_jobs:
            continue
        q_start = str(gdf.at[pos, start_date_col])[:10]
        q_end = str(gdf.at[pos, end_date_col])[:10]
        overlapping = [(jid, js, je) for jid, js, je in geom_jobs if js <= q_end and je >= q_start]
        if not overlapping:
            continue
        job_ids = [jid for jid, _, _ in overlapping]
        covered = [(js, je) for _, js, je in overlapping]
        gaps = _compute_gaps(q_start, q_end, covered)
        result[int(pos)] = (job_ids, gaps)

    return result


# ---------------------------------------------------------------------------
# PostGIS — write
# ---------------------------------------------------------------------------


def _store_sits_pg(
    engine: sa.Engine,
    df: pd.DataFrame,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> int | None:
    if df.empty:
        return None

    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    job_hash = _compute_job_hash(geom_hash, params_hash, start_date, end_date)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    rx, ry = _repr_point(geometry)
    gtype = _geom_type_str(geometry)
    h3_coarse, h3_fine = _compute_h3_for_point(rx, ry)

    with engine.begin() as conn:
        conn.execute(sa.text(
            "INSERT INTO geometries "
            "(geom_hash, geometry, repr_point_x, repr_point_y, geom_type, h3_coarse, h3_fine) "
            "VALUES (:h, ST_GeomFromWKB(:wkb, 4326), :rx, :ry, :gt, :hc, :hf) "
            "ON CONFLICT (geom_hash) DO NOTHING"
        ), {"h": geom_hash, "wkb": geometry.wkb, "rx": rx, "ry": ry, "gt": gtype, "hc": h3_coarse, "hf": h3_fine})

        _geom_row = conn.execute(
            sa.text("SELECT id FROM geometries WHERE geom_hash = :h"), {"h": geom_hash}
        ).fetchone()
        assert _geom_row is not None
        geom_id: int = _geom_row[0]

        conn.execute(sa.text("""
            INSERT INTO sits_jobs
              (job_hash, geometry_id, satellite_short_name, params_hash, reducers,
               subsampling_max_pixels, start_date, end_date, fetched_at)
            VALUES (:jh, :gid, :sat, :ph, :red, :sub, :sd, :ed, :now)
            ON CONFLICT (geometry_id, satellite_short_name, params_hash, start_date, end_date) DO NOTHING
        """), {
            "jh": job_hash, "gid": geom_id, "sat": satellite.shortName, "ph": params_hash,
            "red": json.dumps(sorted(reducers)) if reducers else None,
            "sub": subsampling_max_pixels, "sd": start_date, "ed": end_date,
            "now": datetime.now(UTC).isoformat(),
        })

        _job_row = conn.execute(sa.text("""
            SELECT id FROM sits_jobs
            WHERE geometry_id = :gid AND satellite_short_name = :sat
              AND params_hash = :ph AND start_date = :sd AND end_date = :ed
        """), {"gid": geom_id, "sat": satellite.shortName, "ph": params_hash, "sd": start_date, "ed": end_date}).fetchone()
        assert _job_row is not None
        job_id: int = _job_row[0]

        already_exists = conn.execute(
            sa.text(f'SELECT 1 FROM "{table_name}" WHERE job_id = :jid LIMIT 1'), {"jid": job_id}
        ).fetchone()
        if already_exists:
            return job_id

        present_band_cols = [c for c in band_cols if c in df.columns]
        col_names_sql = ", ".join(['"job_id"', '"timestamp"', *[f'"{c}"' for c in present_band_cols]])
        placeholders_sql = ", ".join([":job_id", ":timestamp", *[f":band_{i}" for i in range(len(present_band_cols))]])
        ts_col = "timestamp" if "timestamp" in df.columns else None

        rows_to_insert = []
        for raw in df.to_dict("records"):
            record: dict[str, Any] = {
                "job_id": job_id,
                "timestamp": str(raw[ts_col]) if ts_col and pd.notna(raw[ts_col]) else None,
            }
            for i, c in enumerate(present_band_cols):
                v = raw[c]
                record[f"band_{i}"] = None if pd.isna(v) else float(v)
            rows_to_insert.append(record)

        if rows_to_insert:
            conn.execute(
                sa.text(f'INSERT INTO "{table_name}" ({col_names_sql}) VALUES ({placeholders_sql})'),
                rows_to_insert,
            )

    return job_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_sits_with_gaps(
    engine: CacheEngine,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> tuple[pl.DataFrame, list[tuple[str, str]]]:
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _fetch_sits_with_gaps_duck(engine, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)
    return _fetch_sits_with_gaps_pg(engine, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)


def fetch_sits(
    engine: CacheEngine,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> pl.DataFrame | None:
    df, gaps = fetch_sits_with_gaps(engine, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)
    if gaps:
        return None
    return df if not df.is_empty() else pl.DataFrame()


def fetch_sits_by_job_ids(
    engine: CacheEngine,
    satellite: AbstractSatellite,
    job_ids: list[int],
) -> dict[int, pl.DataFrame]:
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _fetch_sits_by_jids_duck(engine, satellite, job_ids)
    return _fetch_sits_by_jids_pg(engine, satellite, job_ids)


def fetch_sits_batch_coverage(
    engine: CacheEngine,
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    start_date_col: str,
    end_date_col: str,
) -> dict[int, tuple[list[int], list[tuple[str, str]]]]:
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _fetch_sits_batch_coverage_duck(engine, gdf, satellite, reducers, subsampling_max_pixels, start_date_col, end_date_col)
    return _fetch_sits_batch_coverage_pg(engine, gdf, satellite, reducers, subsampling_max_pixels, start_date_col, end_date_col)


def store_sits(
    engine: CacheEngine,
    df: pd.DataFrame,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> int | None:
    if df.empty:
        return None
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _store_sits_duck(engine, df, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)
    return _store_sits_pg(engine, df, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)


def store_sits_polars(
    engine: CacheEngine,
    df: pl.DataFrame,
    geometry: Any,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> int | None:
    if df.is_empty():
        return None
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _store_sits_duck_polars(engine, df, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)
    return _store_sits_pg(engine, df.to_pandas(), geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def print_cache_status(engine: CacheEngine | None = None) -> None:
    eng = engine or get_engine()
    if eng is None:
        print("[AgriGEE cache] Cache not initialised.")
        return

    if isinstance(eng, duckdb.DuckDBPyConnection):
        total_row = eng.execute("SELECT COUNT(*) FROM sits_jobs").fetchone()
        assert total_row is not None
        total_sits: int = int(total_row[0])

        sat_tables = sorted(
            r[0]
            for r in eng.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            if r[0] not in _DUCK_SYSTEM
        )
        images_by_sensor: dict[str, int] = {}
        for tbl in sat_tables:
            cnt_row = eng.execute(f'SELECT COUNT(*) FROM "{tbl}"').fetchone()
            assert cnt_row is not None
            images_by_sensor[tbl] = int(cnt_row[0])
        backend = f"DuckDB ({DEFAULT_DB_PATH})"
    else:
        with eng.connect() as conn:
            _count_row = conn.execute(sa.text("SELECT COUNT(*) FROM sits_jobs")).fetchone()
            assert _count_row is not None
            total_sits = _count_row[0]
            rows = conn.execute(sa.text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )).fetchall()
            sat_tables = sorted(r[0] for r in rows if r[0] not in _PG_SYSTEM)
            images_by_sensor = {}
            for tbl in sat_tables:
                _tbl_row = conn.execute(sa.text(f'SELECT COUNT(*) FROM "{tbl}"')).fetchone()
                assert _tbl_row is not None
                images_by_sensor[tbl] = _tbl_row[0]
        backend = f"PostGIS ({_PG_DB_NAME})"

    total_images = sum(images_by_sensor.values())
    lines = [
        f"[AgriGEE cache] Status  [{backend}]",
        f"  Total images : {total_images}",
        f"  Total SITS   : {total_sits}",
        "  Images per sensor:",
        *[f"    {sensor:<30} {count:>7}" for sensor, count in images_by_sensor.items()],
    ]
    print("\n".join(lines))


def init_cache(db_path: pathlib.Path = DEFAULT_DB_PATH) -> CacheEngine:
    global _duck_conn, _pg_engine

    from agrigee_lite.sat import (
        ANADEM,
        CopernicusDEM,
        HLSLandsat,
        HLSSentinel2,
        Landsat5,
        Landsat7,
        Landsat8,
        Landsat9,
        MapBiomas,
        Modis8Days,
        ModisDaily,
        NAIP,
        PALSAR2ScanSAR,
        SatelliteEmbedding,
        Sentinel1GRD,
        Sentinel2,
        WRBSoilClasses,
    )

    satellites: list[AbstractSatellite] = [
        Sentinel2(), Sentinel2(use_sr=False),
        Sentinel1GRD(), Sentinel1GRD(ascending=False),
        Landsat5(), Landsat7(), Landsat8(), Landsat9(),
        ModisDaily(), Modis8Days(),
        NAIP(), HLSSentinel2(), HLSLandsat(),
        PALSAR2ScanSAR(), CopernicusDEM(), ANADEM(),
        WRBSoilClasses(), MapBiomas(), SatelliteEmbedding(),
    ]

    if _pg_env_set():
        logger.info("AgriGEE cache: using PostGIS backend (host=%s)", os.environ["AGRIGEE_PG_HOST"])
        engine_pg = _make_pg_engine()
        with engine_pg.begin() as conn:
            _ensure_postgis_extension(conn)
            _ensure_geometries_table_pg(conn)
            _ensure_sits_jobs_table_pg(conn)
            for sat in satellites:
                _ensure_satellite_table_pg(conn, sat.shortName, _get_band_columns(sat))
        _pg_engine = engine_pg
        return engine_pg
    else:
        logger.info("AgriGEE cache: using DuckDB backend (%s)", db_path)
        conn = _make_duckdb_conn(db_path)
        _ensure_schema_duck(conn)
        for sat in satellites:
            _ensure_sat_table_duck(conn, sat.shortName, _get_band_columns(sat))
        _duck_conn = conn
        return conn


def _delete_dir_contents(directory: pathlib.Path) -> int:
    count = 0
    for f in directory.rglob("*"):
        if f.is_file():
            f.unlink()
            count += 1
    for d in sorted(directory.rglob("*"), reverse=True):
        if d.is_dir():
            with contextlib.suppress(OSError):
                d.rmdir()
    return count


def _clear_sits_db(removed: list[str]) -> None:
    global _duck_conn
    eng = get_engine()
    if eng is not None and isinstance(eng, sa.Engine):
        with eng.begin() as conn:
            conn.execute(sa.text("DROP SCHEMA public CASCADE"))
            conn.execute(sa.text("CREATE SCHEMA public"))
        removed.append("PostGIS schema (agrigeelite)")
    else:
        if _duck_conn is not None:
            _duck_conn.close()
            _duck_conn = None
        db_file = DEFAULT_DB_PATH
        if db_file.exists():
            db_file.unlink()
            removed.append(str(db_file))
        init_cache(db_file)


def clear_cache(
    sits_db: bool = True,
    sits_files: bool = True,
    image_files: bool = True,
) -> None:
    cache_root = pathlib.Path.home() / ".cache" / "agrigee_lite"
    removed: list[str] = []

    if sits_db:
        _clear_sits_db(removed)

    if sits_files:
        sits_dir = cache_root / "sits"
        if sits_dir.exists():
            count = _delete_dir_contents(sits_dir)
            removed.append(f"{sits_dir} ({count} files)")

    if image_files:
        images_dir = cache_root / "images"
        if images_dir.exists():
            count = _delete_dir_contents(images_dir)
            removed.append(f"{images_dir} ({count} files)")

    if removed:
        print("[AgriGEE cache] Cleared:")
        for entry in removed:
            print(f"  {entry}")
    else:
        print("[AgriGEE cache] Nothing to clear.")
