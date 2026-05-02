"""DuckDB / PostGIS SITS cache.

Backend selection
-----------------
By default the cache uses a local DuckDB file at
``~/.cache/agrigee_lite/sits_cache.duckdb``.

If the following environment variables are **all** set, a PostGIS database is
used instead (the DuckDB file is ignored):

    AGRIGEE_PG_HOST      – PostgreSQL host
    AGRIGEE_PG_USER      – PostgreSQL user
    AGRIGEE_PG_PASSWORD  – PostgreSQL password
    AGRIGEE_PG_PORT      – PostgreSQL port (optional, default 5432)

The PostGIS backend always uses a database named ``agrigeelite``, which is
created automatically if it does not exist.

Schema
------
geometries
    id        BIGINT  PRIMARY KEY (IDENTITY / SERIAL)
    geom_hash TEXT    NOT NULL UNIQUE   -- SHA-1 of WKB bytes; O(1) dedup
    geometry  BLOB / geometry(4326)     -- WGS-84 WKB

requests
    id                     BIGINT PRIMARY KEY
    geometry_id            FK → geometries(id)
    satellite_short_name   TEXT NOT NULL
    params_hash            TEXT NOT NULL  -- SHA-1(satellite config + reducers + subsampling)
    reducers               TEXT           -- JSON array or NULL
    subsampling_max_pixels REAL NOT NULL
    h3_coarse              TEXT NOT NULL  -- H3 cell used for spatial prefiltering
    h3_fine                TEXT NOT NULL  -- H3 child cell used for spatial prefiltering
    start_date             TEXT NOT NULL  -- ISO-8601
    end_date               TEXT NOT NULL  -- ISO-8601
    fetched_at             TIMESTAMPTZ NOT NULL
    UNIQUE (geometry_id, satellite_short_name, params_hash, start_date, end_date)

<satellite_shortname>  (one table per concrete satellite)
    id         BIGINT PRIMARY KEY
    request_id FK → requests(id)
    timestamp  TIMESTAMPTZ
    <band>     REAL   -- one column per available band
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from datetime import UTC, datetime
from typing import Any

import duckdb
import h3
import pandas as pd
import polars as pl
import sqlalchemy as sa
from sqlalchemy.pool import NullPool

from agrigee_lite.sat.abstract_satellite import AbstractSatellite

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = pathlib.Path.home() / ".cache" / "agrigee_lite" / "sits_cache.duckdb"
_PG_DB_NAME = "agrigeelite"

# Active connections — at most one is non-None at a time.
_duck_conn: duckdb.DuckDBPyConnection | None = None
_pg_engine: sa.Engine | None = None

CacheEngine = duckdb.DuckDBPyConnection | sa.Engine


# ---------------------------------------------------------------------------
# Public engine accessor
# ---------------------------------------------------------------------------


def get_engine() -> CacheEngine | None:
    """Return the active cache connection, or None if cache was not initialised."""
    if _duck_conn is not None:
        return _duck_conn
    return _pg_engine


def _pg_env_set() -> bool:
    return all(k in os.environ for k in ("AGRIGEE_PG_HOST", "AGRIGEE_PG_USER", "AGRIGEE_PG_PASSWORD"))


# ---------------------------------------------------------------------------
# Shared helpers (backend-agnostic)
# ---------------------------------------------------------------------------


def _compute_geom_hash(geometry) -> str:
    return hashlib.sha1(geometry.wkb).hexdigest()  # noqa: S324


compute_geom_hash = _compute_geom_hash


def _compute_h3_cells(geometry, coarse_resolution: int = 5, fine_resolution: int = 8) -> tuple[str, str]:
    centroid = geometry.centroid
    fine_cell = h3.latlng_to_cell(centroid.y, centroid.x, fine_resolution)
    coarse_cell = h3.cell_to_parent(fine_cell, coarse_resolution)
    return coarse_cell, fine_cell


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


def _get_band_columns(satellite: AbstractSatellite) -> list[str]:
    if satellite.availableBands:
        return list(satellite.availableBands.keys())
    if satellite.toDownloadSelectors:
        return [s.split("_", 1)[1] if "_" in s else s for s in satellite.toDownloadSelectors]
    if hasattr(satellite, "classes"):
        return [f"soil_{info['label'].lower()}" for info in getattr(satellite, "classes").values()]
    return []


def _chunked(values: list[Any], size: int) -> list[list[Any]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


# ---------------------------------------------------------------------------
# DuckDB backend — schema
# ---------------------------------------------------------------------------


def _make_duckdb_conn(db_path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(db_path))


def _ensure_schema_duck(conn: duckdb.DuckDBPyConnection) -> None:
    conn.execute("CREATE SEQUENCE IF NOT EXISTS geometries_id_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS geometries (
            id        BIGINT PRIMARY KEY DEFAULT nextval('geometries_id_seq'),
            geom_hash TEXT   NOT NULL UNIQUE,
            geometry  BLOB
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_geom_hash ON geometries(geom_hash)")
    conn.execute("CREATE SEQUENCE IF NOT EXISTS requests_id_seq")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id                     BIGINT PRIMARY KEY DEFAULT nextval('requests_id_seq'),
            geometry_id            BIGINT NOT NULL REFERENCES geometries(id),
            satellite_short_name   TEXT   NOT NULL,
            params_hash            TEXT   NOT NULL,
            reducers               TEXT,
            subsampling_max_pixels DOUBLE NOT NULL,
            h3_coarse              TEXT   NOT NULL DEFAULT '',
            h3_fine                TEXT   NOT NULL DEFAULT '',
            start_date             TEXT   NOT NULL,
            end_date               TEXT   NOT NULL,
            fetched_at             TIMESTAMPTZ NOT NULL,
            UNIQUE (geometry_id, satellite_short_name, params_hash, start_date, end_date)
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_req_sat ON requests(satellite_short_name, params_hash)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_req_h3c ON requests(h3_coarse)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_req_h3f ON requests(h3_fine)")


def _ensure_sat_table_duck(conn: duckdb.DuckDBPyConnection, table_name: str, band_cols: list[str]) -> None:
    if not band_cols:
        return
    seq_name = f"{table_name}_id_seq"
    conn.execute(f"CREATE SEQUENCE IF NOT EXISTS \"{seq_name}\"")
    band_defs = ",\n    ".join(f'"{c}" DOUBLE' for c in band_cols)
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id         BIGINT PRIMARY KEY DEFAULT nextval('"{seq_name}"'),
            request_id BIGINT NOT NULL REFERENCES requests(id),
            timestamp  TIMESTAMPTZ,
            {band_defs}
        )
    """)
    conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_rid" ON "{table_name}"(request_id)')


# ---------------------------------------------------------------------------
# DuckDB backend — reads
# ---------------------------------------------------------------------------


def _fetch_records_by_h3_duck(
    conn: duckdb.DuckDBPyConnection,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    h3_pairs: set[tuple[str, str]],
) -> list[tuple[str, str, str, str, str, int]]:
    """Two-stage H3 lookup: coarse index scan first, fine index narrows result.

    H3 hierarchy guarantees each fine cell belongs to exactly one coarse cell,
    so separate IN clauses on indexed columns produce the same result as tuple-IN
    without the index-unfriendly tuple comparison.
    """
    if not h3_pairs:
        return []

    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    coarse_cells = sorted({p[0] for p in h3_pairs})
    fine_cells = sorted({p[1] for p in h3_pairs})
    c_ph = ", ".join("?" * len(coarse_cells))
    f_ph = ", ".join("?" * len(fine_cells))

    rows = conn.execute(
        f"""
        SELECT g.geom_hash, r.h3_coarse, r.h3_fine, r.start_date, r.end_date, r.id
        FROM requests r
        JOIN geometries g ON r.geometry_id = g.id
        WHERE r.satellite_short_name = ?
          AND r.params_hash = ?
          AND r.h3_coarse IN ({c_ph})
          AND r.h3_fine   IN ({f_ph})
        """,
        [satellite.shortName, params_hash, *coarse_cells, *fine_cells],
    ).fetchall()
    return [(r[0], r[1], r[2], r[3], r[4], int(r[5])) for r in rows]


def _fetch_sits_duck(
    conn: duckdb.DuckDBPyConnection,
    geometry,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> pd.DataFrame | None:
    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    req_row = conn.execute(
        """
        SELECT r.id FROM requests r
        JOIN geometries g ON r.geometry_id = g.id
        WHERE g.geom_hash = ? AND r.satellite_short_name = ?
          AND r.params_hash = ? AND r.start_date = ? AND r.end_date = ?
        """,
        [geom_hash, satellite.shortName, params_hash, start_date, end_date],
    ).fetchone()
    if req_row is None:
        return None

    cols_sql = ", ".join(f'"{c}"' for c in band_cols)
    pl_df = conn.execute(
        f'SELECT timestamp, {cols_sql} FROM "{table_name}" WHERE request_id = ? ORDER BY timestamp',
        [int(req_row[0])],
    ).pl()

    if pl_df.is_empty():
        return None

    df = pl_df.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _fetch_sits_by_rids_duck(
    conn: duckdb.DuckDBPyConnection,
    satellite: AbstractSatellite,
    request_ids: list[int],
) -> dict[int, pd.DataFrame]:
    if not request_ids:
        return {}

    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)
    cols_sql = ", ".join(f'"{c}"' for c in band_cols)
    result: dict[int, pd.DataFrame] = {}

    for chunk in _chunked(list(dict.fromkeys(request_ids)), 400):
        ph = ", ".join("?" * len(chunk))
        pl_df = conn.execute(
            f'SELECT request_id, timestamp, {cols_sql} FROM "{table_name}"'
            f" WHERE request_id IN ({ph}) ORDER BY request_id, timestamp",
            chunk,
        ).pl()

        for sub in pl_df.partition_by("request_id", maintain_order=True):
            rid = int(sub["request_id"][0])
            pd_sub = sub.drop("request_id").to_pandas()
            pd_sub["timestamp"] = pd.to_datetime(pd_sub["timestamp"])
            result[rid] = pd_sub

    return result


# ---------------------------------------------------------------------------
# DuckDB backend — write
# ---------------------------------------------------------------------------


def _store_sits_duck(
    conn: duckdb.DuckDBPyConnection,
    df: pd.DataFrame,
    geometry,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    h3_coarse: str,
    h3_fine: str,
) -> int | None:
    if df.empty:
        return None

    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    conn.begin()
    try:
        conn.execute(
            "INSERT INTO geometries (geom_hash, geometry) VALUES (?, ?) ON CONFLICT (geom_hash) DO NOTHING",
            [geom_hash, geometry.wkb],
        )
        geom_id_row = conn.execute("SELECT id FROM geometries WHERE geom_hash = ?", [geom_hash]).fetchone()
        assert geom_id_row is not None
        geom_id: int = int(geom_id_row[0])

        conn.execute(
            """
            INSERT INTO requests
              (geometry_id, satellite_short_name, params_hash, reducers,
               subsampling_max_pixels, h3_coarse, h3_fine, start_date, end_date, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (geometry_id, satellite_short_name, params_hash, start_date, end_date) DO NOTHING
            """,
            [
                geom_id, satellite.shortName, params_hash,
                json.dumps(sorted(reducers)) if reducers else None,
                subsampling_max_pixels, h3_coarse, h3_fine, start_date, end_date,
                datetime.now(UTC).isoformat(),
            ],
        )
        req_row = conn.execute(
            """SELECT id FROM requests WHERE geometry_id = ? AND satellite_short_name = ?
               AND params_hash = ? AND start_date = ? AND end_date = ?""",
            [geom_id, satellite.shortName, params_hash, start_date, end_date],
        ).fetchone()
        assert req_row is not None
        req_id: int = int(req_row[0])

        already = conn.execute(
            f'SELECT 1 FROM "{table_name}" WHERE request_id = ? LIMIT 1', [req_id]
        ).fetchone()
        if already:
            conn.commit()
            return req_id

        present_band_cols = [c for c in band_cols if c in df.columns]
        ts_col = "timestamp" if "timestamp" in df.columns else None
        col_names = ", ".join(["request_id", "timestamp", *[f'"{c}"' for c in present_band_cols]])
        ph = ", ".join(["?"] * (2 + len(present_band_cols)))

        rows: list[list[Any]] = []
        for raw in df.to_dict("records"):
            ts_val = str(raw[ts_col]) if ts_col and pd.notna(raw[ts_col]) else None
            rows.append([req_id, ts_val, *[None if pd.isna(raw[c]) else float(raw[c]) for c in present_band_cols]])

        if rows:
            conn.executemany(f'INSERT INTO "{table_name}" ({col_names}) VALUES ({ph})', rows)

        conn.commit()
    except Exception:
        conn.rollback()
        raise

    return req_id


# ---------------------------------------------------------------------------
# PostGIS backend — schema (unchanged from original)
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
    conn.execute(
        sa.text("""
        CREATE TABLE IF NOT EXISTS geometries (
            id        SERIAL PRIMARY KEY,
            geom_hash TEXT NOT NULL UNIQUE,
            geometry  geometry(Geometry, 4326)
        )
    """)
    )
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_geometries_geom ON geometries USING GIST (geometry)"))


def _ensure_requests_table_pg(conn: sa.Connection) -> None:
    conn.execute(
        sa.text("""
        CREATE TABLE IF NOT EXISTS requests (
            id                     SERIAL PRIMARY KEY,
            geometry_id            INTEGER NOT NULL REFERENCES geometries(id),
            satellite_short_name   TEXT NOT NULL,
            params_hash            TEXT NOT NULL,
            reducers               TEXT,
            subsampling_max_pixels REAL NOT NULL,
            h3_coarse              TEXT NOT NULL DEFAULT '',
            h3_fine                TEXT NOT NULL DEFAULT '',
            start_date             TEXT NOT NULL,
            end_date               TEXT NOT NULL,
            fetched_at             TIMESTAMPTZ NOT NULL,
            UNIQUE (geometry_id, satellite_short_name, params_hash, start_date, end_date)
        )
    """)
    )
    existing_columns = {
        row[0]
        for row in conn.execute(
            sa.text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'requests'
        """)
        ).fetchall()
    }
    if "h3_coarse" not in existing_columns:
        conn.execute(sa.text("ALTER TABLE requests ADD COLUMN h3_coarse TEXT NOT NULL DEFAULT ''"))
    if "h3_fine" not in existing_columns:
        conn.execute(sa.text("ALTER TABLE requests ADD COLUMN h3_fine TEXT NOT NULL DEFAULT ''"))
    conn.execute(
        sa.text(
            "CREATE INDEX IF NOT EXISTS idx_requests_lookup "
            "ON requests (geometry_id, satellite_short_name, params_hash)"
        )
    )
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_requests_h3c ON requests (h3_coarse)"))
    conn.execute(sa.text("CREATE INDEX IF NOT EXISTS idx_requests_h3f ON requests (h3_fine)"))


def _ensure_satellite_table_pg(conn: sa.Connection, table_name: str, band_columns: list[str]) -> None:
    if not band_columns:
        return
    band_cols_sql = ",\n    ".join(f'"{col}" REAL' for col in band_columns)
    conn.execute(
        sa.text(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id         SERIAL PRIMARY KEY,
            request_id INTEGER NOT NULL REFERENCES requests(id),
            timestamp  TIMESTAMPTZ,
            {band_cols_sql}
        )
    """)
    )
    conn.execute(sa.text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_rid" ON "{table_name}" (request_id)'))
    conn.execute(sa.text(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_ts"  ON "{table_name}" (timestamp)'))


# ---------------------------------------------------------------------------
# PostGIS backend — reads
# ---------------------------------------------------------------------------


def _fetch_records_by_h3_pg(
    engine: sa.Engine,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    h3_pairs: set[tuple[str, str]],
) -> list[tuple[str, str, str, str, str, int]]:
    if not h3_pairs:
        return []

    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    coarse_cells = sorted({p[0] for p in h3_pairs})
    fine_cells = sorted({p[1] for p in h3_pairs})

    with engine.connect() as conn:
        rows = conn.execute(
            sa.text("""
                SELECT g.geom_hash, r.h3_coarse, r.h3_fine, r.start_date, r.end_date, r.id
                FROM requests r
                JOIN geometries g ON r.geometry_id = g.id
                WHERE r.satellite_short_name = :sat
                  AND r.params_hash = :ph
                  AND r.h3_coarse = ANY(:coarse)
                  AND r.h3_fine   = ANY(:fine)
            """),
            {"sat": satellite.shortName, "ph": params_hash, "coarse": coarse_cells, "fine": fine_cells},
        ).fetchall()
    return [(r[0], r[1], r[2], r[3], r[4], int(r[5])) for r in rows]


def _fetch_sits_pg(
    engine: sa.Engine,
    geometry,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> pd.DataFrame | None:
    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    with engine.connect() as conn:
        geom_row = conn.execute(
            sa.text("SELECT id FROM geometries WHERE geom_hash = :h"), {"h": geom_hash}
        ).fetchone()
        if geom_row is None:
            return None
        req_row = conn.execute(
            sa.text("""
                SELECT id FROM requests
                WHERE geometry_id = :gid AND satellite_short_name = :sat
                  AND params_hash = :ph AND start_date = :sd AND end_date = :ed
            """),
            {"gid": geom_row[0], "sat": satellite.shortName, "ph": params_hash, "sd": start_date, "ed": end_date},
        ).fetchone()
        if req_row is None:
            return None

        cols_sql = ", ".join(f'"{c}"' for c in ["timestamp", *band_cols])
        rows = conn.execute(
            sa.text(f'SELECT {cols_sql} FROM "{table_name}" WHERE request_id = :rid ORDER BY timestamp'),
            {"rid": req_row[0]},
        ).fetchall()

    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", *band_cols])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def _fetch_sits_by_rids_pg(
    engine: sa.Engine,
    satellite: AbstractSatellite,
    request_ids: list[int],
) -> dict[int, pd.DataFrame]:
    if not request_ids:
        return {}

    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)
    cols_sql = ", ".join(f'"{c}"' for c in ["timestamp", *band_cols])
    result: dict[int, pd.DataFrame] = {}
    unique_request_ids = list(dict.fromkeys(request_ids))

    with engine.connect() as conn:
        for chunk in _chunked(unique_request_ids, 400):
            placeholders = ", ".join(f":id{i}" for i in range(len(chunk)))
            params: dict[str, int] = {f"id{i}": rid for i, rid in enumerate(chunk)}
            rows = conn.execute(
                sa.text(
                    f'SELECT request_id, {cols_sql} FROM "{table_name}"'
                    f" WHERE request_id IN ({placeholders})"
                    f" ORDER BY request_id, timestamp"
                ),
                params,
            ).fetchall()

            current_id: int | None = None
            current_rows: list[Any] = []
            for row in rows:
                rid = int(row[0])
                if rid != current_id:
                    if current_id is not None and current_rows:
                        df = pd.DataFrame(current_rows, columns=["timestamp", *band_cols])
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        result[current_id] = df
                    current_id = rid
                    current_rows = []
                current_rows.append(row[1:])
            if current_id is not None and current_rows:
                df = pd.DataFrame(current_rows, columns=["timestamp", *band_cols])
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                result[current_id] = df

    return result


# ---------------------------------------------------------------------------
# PostGIS backend — write
# ---------------------------------------------------------------------------


def _store_sits_pg(
    engine: sa.Engine,
    df: pd.DataFrame,
    geometry,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    h3_coarse: str,
    h3_fine: str,
) -> int | None:
    if df.empty:
        return None

    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    with engine.begin() as conn:
        conn.execute(
            sa.text(
                "INSERT INTO geometries (geom_hash, geometry) VALUES (:h, ST_GeomFromWKB(:wkb, 4326))"
                " ON CONFLICT (geom_hash) DO NOTHING"
            ),
            {"h": geom_hash, "wkb": geometry.wkb},
        )
        _geom_row = conn.execute(
            sa.text("SELECT id FROM geometries WHERE geom_hash = :h"), {"h": geom_hash}
        ).fetchone()
        assert _geom_row is not None
        geom_id: int = _geom_row[0]

        conn.execute(
            sa.text("""
                INSERT INTO requests
                  (geometry_id, satellite_short_name, params_hash, reducers,
                   subsampling_max_pixels, h3_coarse, h3_fine, start_date, end_date, fetched_at)
                VALUES (:gid, :sat, :ph, :red, :sub, :h3c, :h3f, :sd, :ed, :now)
                ON CONFLICT (geometry_id, satellite_short_name, params_hash, start_date, end_date) DO NOTHING
            """),
            {
                "gid": geom_id, "sat": satellite.shortName, "ph": params_hash,
                "red": json.dumps(sorted(reducers)) if reducers else None,
                "sub": subsampling_max_pixels, "h3c": h3_coarse, "h3f": h3_fine,
                "sd": start_date, "ed": end_date, "now": datetime.now(UTC).isoformat(),
            },
        )
        _req_row = conn.execute(
            sa.text("""
                SELECT id FROM requests
                WHERE geometry_id = :gid AND satellite_short_name = :sat
                  AND params_hash = :ph AND start_date = :sd AND end_date = :ed
            """),
            {"gid": geom_id, "sat": satellite.shortName, "ph": params_hash, "sd": start_date, "ed": end_date},
        ).fetchone()
        assert _req_row is not None
        req_id: int = _req_row[0]

        already_exists = conn.execute(
            sa.text(f'SELECT 1 FROM "{table_name}" WHERE request_id = :rid LIMIT 1'), {"rid": req_id}
        ).fetchone()
        if already_exists:
            return req_id

        present_band_cols = [c for c in band_cols if c in df.columns]
        col_names_sql = ", ".join(['"request_id"', '"timestamp"', *[f'"{c}"' for c in present_band_cols]])
        placeholders_sql = ", ".join([":request_id", ":timestamp", *[f":band_{i}" for i in range(len(present_band_cols))]])
        ts_col = "timestamp" if "timestamp" in df.columns else None

        rows_to_insert = []
        for raw in df.to_dict("records"):
            record: dict[str, Any] = {
                "request_id": req_id,
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

    return req_id


# ---------------------------------------------------------------------------
# Public API — dispatch to DuckDB or PostGIS
# ---------------------------------------------------------------------------


def fetch_sits_cache_records_by_h3(
    engine: CacheEngine,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    h3_pairs: set[tuple[str, str]],
) -> list[tuple[str, str, str, str, str, int]]:
    """Return cached requests filtered by H3 coarse+fine pairs.

    Uses two separate indexed IN clauses (one per H3 resolution) instead of
    a tuple-IN, which lets the query planner use each column's index independently.
    H3 hierarchy guarantees no false positives: every fine cell belongs to exactly
    one coarse cell, so the cross-product of both sets matches only valid pairs.
    """
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _fetch_records_by_h3_duck(engine, satellite, reducers, subsampling_max_pixels, h3_pairs)
    return _fetch_records_by_h3_pg(engine, satellite, reducers, subsampling_max_pixels, h3_pairs)


def fetch_sits(
    engine: CacheEngine,
    geometry,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> pd.DataFrame | None:
    """Return a cached SITS DataFrame, or None on cache miss."""
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _fetch_sits_duck(engine, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)
    return _fetch_sits_pg(engine, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)


def fetch_sits_by_request_ids(
    engine: CacheEngine,
    satellite: AbstractSatellite,
    request_ids: list[int],
) -> dict[int, pd.DataFrame]:
    """Batch-fetch cached SITS rows for a list of request_ids."""
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _fetch_sits_by_rids_duck(engine, satellite, request_ids)
    return _fetch_sits_by_rids_pg(engine, satellite, request_ids)


def store_sits(
    engine: CacheEngine,
    df: pd.DataFrame,
    geometry,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    h3_coarse: str | None = None,
    h3_fine: str | None = None,
) -> int | None:
    """Persist a SITS DataFrame into the cache."""
    if df.empty:
        return None
    if h3_coarse is None or h3_fine is None:
        h3_coarse, h3_fine = _compute_h3_cells(geometry)
    if isinstance(engine, duckdb.DuckDBPyConnection):
        return _store_sits_duck(engine, df, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels, h3_coarse, h3_fine)
    return _store_sits_pg(engine, df, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels, h3_coarse, h3_fine)


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def print_cache_status(engine: CacheEngine | None = None) -> None:
    """Print a summary of the current cache state to stdout."""
    eng = engine or get_engine()
    if eng is None:
        print("[AgriGEE cache] Cache not initialised.")
        return

    _PG_SYSTEM = {"geometries", "requests", "spatial_ref_sys", "geometry_columns"}
    _DUCK_SYSTEM = {"geometries", "requests"}

    if isinstance(eng, duckdb.DuckDBPyConnection):
        total_sits_row = eng.execute("SELECT COUNT(*) FROM requests").fetchone()
        assert total_sits_row is not None
        total_sits: int = int(total_sits_row[0])

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
            _count_row = conn.execute(sa.text("SELECT COUNT(*) FROM requests")).fetchone()
            assert _count_row is not None
            total_sits = _count_row[0]
            rows = conn.execute(
                sa.text(
                    "SELECT table_name FROM information_schema.tables "
                    "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
                )
            ).fetchall()
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
    """Initialise the SITS cache (DuckDB local or PostGIS remote).

    If ``AGRIGEE_PG_HOST``, ``AGRIGEE_PG_USER``, and ``AGRIGEE_PG_PASSWORD``
    are set, PostGIS is used. Otherwise a local DuckDB file is created at
    *db_path* (default ``~/.cache/agrigee_lite/sits_cache.duckdb``).

    All schema operations are idempotent — safe to call on every startup.
    """
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
            _ensure_requests_table_pg(conn)
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


def clear_cache(
    sits_db: bool = True,
    sits_files: bool = True,
    image_files: bool = True,
) -> None:
    """Delete all locally cached data produced by AgriGEE.lite."""
    global _duck_conn, _pg_engine

    cache_root = pathlib.Path.home() / ".cache" / "agrigee_lite"
    removed: list[str] = []

    if sits_db:
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

    if sits_files:
        sits_dir = cache_root / "sits"
        if sits_dir.exists():
            count = 0
            for f in sits_dir.rglob("*"):
                if f.is_file():
                    f.unlink()
                    count += 1
            for d in sorted(sits_dir.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except OSError:
                        pass
            removed.append(f"{sits_dir} ({count} files)")

    if image_files:
        images_dir = cache_root / "images"
        if images_dir.exists():
            count = 0
            for f in images_dir.rglob("*"):
                if f.is_file():
                    f.unlink()
                    count += 1
            for d in sorted(images_dir.rglob("*"), reverse=True):
                if d.is_dir():
                    try:
                        d.rmdir()
                    except OSError:
                        pass
            removed.append(f"{images_dir} ({count} files)")

    if removed:
        print("[AgriGEE cache] Cleared:")
        for entry in removed:
            print(f"  {entry}")
    else:
        print("[AgriGEE cache] Nothing to clear.")


# ---------------------------------------------------------------------------
# Legacy alias kept for any external code that imported fetch_sits_cache_index
# ---------------------------------------------------------------------------


def fetch_sits_cache_index(
    engine: CacheEngine,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> dict[tuple[str, str, str], int]:
    """Return a dict mapping (geom_hash, start_date, end_date) → request_id."""
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    if isinstance(engine, duckdb.DuckDBPyConnection):
        rows = engine.execute(
            """
            SELECT g.geom_hash, r.start_date, r.end_date, r.id
            FROM requests r JOIN geometries g ON r.geometry_id = g.id
            WHERE r.satellite_short_name = ? AND r.params_hash = ?
            """,
            [satellite.shortName, params_hash],
        ).fetchall()
    else:
        with engine.connect() as conn:
            rows = conn.execute(
                sa.text("""
                    SELECT g.geom_hash, r.start_date, r.end_date, r.id
                    FROM requests r JOIN geometries g ON r.geometry_id = g.id
                    WHERE r.satellite_short_name = :sat AND r.params_hash = :ph
                """),
                {"sat": satellite.shortName, "ph": params_hash},
            ).fetchall()
    return {(r[0], r[1], r[2]): int(r[3]) for r in rows}
