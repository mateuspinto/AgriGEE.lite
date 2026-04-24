"""SpatiaLite / PostGIS SITS cache.

Backend selection
-----------------
By default the cache uses a local SpatiaLite file at
``~/.cache/agrigee_lite/sits_cache.db``.

If the following environment variables are **all** set, a PostGIS database is
used instead (the SpatiaLite file is ignored):

    AGRIGEE_PG_HOST      – PostgreSQL host
    AGRIGEE_PG_USER      – PostgreSQL user
    AGRIGEE_PG_PASSWORD  – PostgreSQL password
    AGRIGEE_PG_PORT      – PostgreSQL port (optional, default 5432)

The PostGIS backend always uses a database named ``agrigeelite``, which is
created automatically if it does not exist.

Schema
------
geometries
    id        INTEGER / SERIAL PRIMARY KEY
    geom_hash TEXT NOT NULL UNIQUE   -- SHA-1 of WKB bytes; O(1) dedup lookup
    geometry  GEOMETRY(4326)         -- WGS-84

requests
    id                     INTEGER / SERIAL PRIMARY KEY
    geometry_id            FK → geometries(id)
    satellite_short_name   TEXT NOT NULL
    params_hash            TEXT NOT NULL  -- SHA-1(satellite config + reducers + subsampling)
    reducers               TEXT           -- JSON array or NULL
    subsampling_max_pixels REAL NOT NULL
    start_date             TEXT NOT NULL  -- ISO-8601
    end_date               TEXT NOT NULL  -- ISO-8601
    fetched_at             DATETIME / TIMESTAMPTZ NOT NULL
    UNIQUE (geometry_id, satellite_short_name, params_hash, start_date, end_date)

<satellite_shortname>  (one table per concrete satellite)
    id         INTEGER / SERIAL PRIMARY KEY
    request_id FK → requests(id)
    timestamp  DATETIME / TIMESTAMPTZ
    <band>     REAL   -- one column per available band
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pathlib
from datetime import UTC, datetime

import pandas as pd
import sqlalchemy as sa
from sqlalchemy import event
from sqlalchemy.pool import NullPool

from agrigee_lite.sat.abstract_satellite import AbstractSatellite

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = pathlib.Path.home() / ".cache" / "agrigee_lite" / "sits_cache.db"
_SPATIALITE_EXT = "mod_spatialite"
_PG_DB_NAME = "agrigeelite"

_cache_engine: sa.Engine | None = None


def get_engine() -> sa.Engine | None:
    """Return the active cache engine, or None if cache was not initialised."""
    return _cache_engine


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------


def _load_spatialite(dbapi_conn, _record) -> None:
    dbapi_conn.enable_load_extension(True)
    dbapi_conn.load_extension(_SPATIALITE_EXT)
    dbapi_conn.enable_load_extension(False)


def _make_spatialite_engine(db_path: pathlib.Path) -> sa.Engine:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    engine = sa.create_engine(f"sqlite:///{db_path}", poolclass=NullPool)
    event.listen(engine, "connect", _load_spatialite)
    return engine


def _make_pg_engine() -> sa.Engine:
    """Connect to (or create) the PostGIS ``agrigeelite`` database."""
    host = os.environ["AGRIGEE_PG_HOST"]
    user = os.environ["AGRIGEE_PG_USER"]
    password = os.environ["AGRIGEE_PG_PASSWORD"]
    port = os.environ.get("AGRIGEE_PG_PORT", "5432")

    def _url(db: str) -> str:
        return f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    # Create the database if it doesn't exist (must run outside a transaction)
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


def _pg_env_set() -> bool:
    return all(k in os.environ for k in ("AGRIGEE_PG_HOST", "AGRIGEE_PG_USER", "AGRIGEE_PG_PASSWORD"))


# ---------------------------------------------------------------------------
# Schema creation – SpatiaLite
# ---------------------------------------------------------------------------


def _ensure_spatial_metadata(conn: sa.Connection) -> None:
    exists = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys'")
    ).fetchone()
    if exists is None:
        conn.execute(sa.text("SELECT InitSpatialMetaData(1)"))


def _ensure_geometries_table_sl(conn: sa.Connection) -> None:
    exists = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='geometries'")
    ).fetchone()
    if exists is not None:
        return
    conn.execute(sa.text("""
        CREATE TABLE geometries (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            geom_hash TEXT NOT NULL UNIQUE
        )
    """))
    conn.execute(sa.text("SELECT AddGeometryColumn('geometries', 'geometry', 4326, 'GEOMETRY', 'XY')"))
    conn.execute(sa.text("SELECT CreateSpatialIndex('geometries', 'geometry')"))


def _ensure_requests_table_sl(conn: sa.Connection) -> None:
    exists = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='requests'")
    ).fetchone()
    if exists is not None:
        return
    conn.execute(sa.text("""
        CREATE TABLE requests (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            geometry_id            INTEGER NOT NULL REFERENCES geometries(id),
            satellite_short_name   TEXT NOT NULL,
            params_hash            TEXT NOT NULL,
            reducers               TEXT,
            subsampling_max_pixels REAL NOT NULL,
            start_date             TEXT NOT NULL,
            end_date               TEXT NOT NULL,
            fetched_at             DATETIME NOT NULL,
            UNIQUE (geometry_id, satellite_short_name, params_hash, start_date, end_date)
        )
    """))
    conn.execute(sa.text(
        "CREATE INDEX idx_requests_lookup ON requests (geometry_id, satellite_short_name, params_hash)"
    ))


def _ensure_satellite_table_sl(conn: sa.Connection, table_name: str, band_columns: list[str]) -> None:
    if not band_columns:
        return
    exists = conn.execute(
        sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name=:name"),
        {"name": table_name},
    ).fetchone()
    if exists is not None:
        return
    band_cols_sql = ",\n    ".join(f'"{col}" REAL' for col in band_columns)
    conn.execute(sa.text(f"""
        CREATE TABLE "{table_name}" (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER NOT NULL REFERENCES requests(id),
            timestamp  DATETIME,
            {band_cols_sql}
        )
    """))
    conn.execute(sa.text(f'CREATE INDEX "idx_{table_name}_rid" ON "{table_name}" (request_id)'))
    conn.execute(sa.text(f'CREATE INDEX "idx_{table_name}_ts"  ON "{table_name}" (timestamp)'))


# ---------------------------------------------------------------------------
# Schema creation – PostGIS
# ---------------------------------------------------------------------------


def _ensure_postgis_extension(conn: sa.Connection) -> None:
    conn.execute(sa.text("CREATE EXTENSION IF NOT EXISTS postgis"))


def _ensure_geometries_table_pg(conn: sa.Connection) -> None:
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS geometries (
            id        SERIAL PRIMARY KEY,
            geom_hash TEXT NOT NULL UNIQUE,
            geometry  geometry(Geometry, 4326)
        )
    """))
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_geometries_geom ON geometries USING GIST (geometry)"
    ))


def _ensure_requests_table_pg(conn: sa.Connection) -> None:
    conn.execute(sa.text("""
        CREATE TABLE IF NOT EXISTS requests (
            id                     SERIAL PRIMARY KEY,
            geometry_id            INTEGER NOT NULL REFERENCES geometries(id),
            satellite_short_name   TEXT NOT NULL,
            params_hash            TEXT NOT NULL,
            reducers               TEXT,
            subsampling_max_pixels REAL NOT NULL,
            start_date             TEXT NOT NULL,
            end_date               TEXT NOT NULL,
            fetched_at             TIMESTAMPTZ NOT NULL,
            UNIQUE (geometry_id, satellite_short_name, params_hash, start_date, end_date)
        )
    """))
    conn.execute(sa.text(
        "CREATE INDEX IF NOT EXISTS idx_requests_lookup "
        "ON requests (geometry_id, satellite_short_name, params_hash)"
    ))


def _ensure_satellite_table_pg(conn: sa.Connection, table_name: str, band_columns: list[str]) -> None:
    if not band_columns:
        return
    band_cols_sql = ",\n    ".join(f'"{col}" REAL' for col in band_columns)
    conn.execute(sa.text(f"""
        CREATE TABLE IF NOT EXISTS "{table_name}" (
            id         SERIAL PRIMARY KEY,
            request_id INTEGER NOT NULL REFERENCES requests(id),
            timestamp  TIMESTAMPTZ,
            {band_cols_sql}
        )
    """))
    conn.execute(sa.text(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_rid" ON "{table_name}" (request_id)'
    ))
    conn.execute(sa.text(
        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_ts" ON "{table_name}" (timestamp)'
    ))


# ---------------------------------------------------------------------------
# Satellite metadata helpers
# ---------------------------------------------------------------------------


def _get_band_columns(satellite: AbstractSatellite) -> list[str]:
    """Return the DataFrame column names that a satellite produces (after prepare_output_df)."""
    if satellite.availableBands:
        return list(satellite.availableBands.keys())
    if satellite.toDownloadSelectors:
        return [s.split("_", 1)[1] if "_" in s else s for s in satellite.toDownloadSelectors]
    # WRBSoilClasses: columns built at compute-time from the classes dict
    if hasattr(satellite, "classes"):
        return [f"soil_{info['label'].lower()}" for info in getattr(satellite, "classes").values()]
    return []


def _compute_geom_hash(geometry) -> str:
    return hashlib.sha1(geometry.wkb).hexdigest()  # noqa: S324


def _compute_params_hash(
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> str:
    """Hash that uniquely identifies request parameters affecting the output data."""
    obj = {
        "satellite": satellite.log_dict(),
        "reducers": sorted(reducers) if reducers else None,
        "subsampling_max_pixels": subsampling_max_pixels,
    }
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()  # noqa: S324


# ---------------------------------------------------------------------------
# Public cache API
# ---------------------------------------------------------------------------


def fetch_sits(
    engine: sa.Engine,
    geometry,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> pd.DataFrame | None:
    """Return a cached SITS DataFrame, or None if the request is not in the cache.

    Parameters
    ----------
    engine:
        Cache engine returned by :func:`init_cache` (or :func:`get_engine`).
    geometry:
        Shapely geometry used as the spatial key.
    start_date, end_date:
        ISO-8601 date strings (``"YYYY-MM-DD"``).
    satellite:
        Satellite configuration object.
    reducers:
        Set of reducer names, or None.
    subsampling_max_pixels:
        Subsampling parameter used during the original request.

    Returns
    -------
    pd.DataFrame or None
        Cached data (``timestamp`` + band columns), or ``None`` on a cache miss.
    """
    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    with engine.connect() as conn:
        geom_row = conn.execute(
            sa.text("SELECT id FROM geometries WHERE geom_hash = :h"),
            {"h": geom_hash},
        ).fetchone()
        if geom_row is None:
            return None

        req_row = conn.execute(
            sa.text("""
                SELECT id FROM requests
                WHERE geometry_id          = :gid
                  AND satellite_short_name = :sat
                  AND params_hash          = :ph
                  AND start_date           = :sd
                  AND end_date             = :ed
            """),
            {
                "gid": geom_row[0],
                "sat": satellite.shortName,
                "ph": params_hash,
                "sd": start_date,
                "ed": end_date,
            },
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


def store_sits(
    engine: sa.Engine,
    df: pd.DataFrame,
    geometry,
    start_date: str,
    end_date: str,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
) -> None:
    """Persist a SITS DataFrame into the cache.

    Parameters
    ----------
    engine:
        Cache engine.
    df:
        DataFrame to cache (``timestamp`` + band columns, no ``original_index``).
    geometry:
        Shapely geometry associated with the data.
    start_date, end_date:
        ISO-8601 date strings.
    satellite:
        Satellite configuration object.
    reducers:
        Reducers used during download.
    subsampling_max_pixels:
        Subsampling parameter used during download.
    """
    if df.empty:
        return

    is_pg = engine.dialect.name == "postgresql"
    geom_hash = _compute_geom_hash(geometry)
    params_hash = _compute_params_hash(satellite, reducers, subsampling_max_pixels)
    table_name = satellite.shortName
    band_cols = _get_band_columns(satellite)

    if is_pg:
        geom_fn = "ST_GeomFromWKB(:wkb, 4326)"
        ignore_geom = "INSERT INTO geometries (geom_hash, geometry) VALUES (:h, ST_GeomFromWKB(:wkb, 4326)) ON CONFLICT (geom_hash) DO NOTHING"
        ignore_req = """
            INSERT INTO requests
              (geometry_id, satellite_short_name, params_hash, reducers,
               subsampling_max_pixels, start_date, end_date, fetched_at)
            VALUES (:gid, :sat, :ph, :red, :sub, :sd, :ed, :now)
            ON CONFLICT (geometry_id, satellite_short_name, params_hash, start_date, end_date) DO NOTHING
        """
    else:
        geom_fn = "GeomFromWKB(:wkb, 4326)"  # noqa: F841 (kept for symmetry)
        ignore_geom = "INSERT OR IGNORE INTO geometries (geom_hash, geometry) VALUES (:h, GeomFromWKB(:wkb, 4326))"
        ignore_req = """
            INSERT OR IGNORE INTO requests
              (geometry_id, satellite_short_name, params_hash, reducers,
               subsampling_max_pixels, start_date, end_date, fetched_at)
            VALUES (:gid, :sat, :ph, :red, :sub, :sd, :ed, :now)
        """

    with engine.begin() as conn:
        # --- geometry ---
        conn.execute(sa.text(ignore_geom), {"h": geom_hash, "wkb": geometry.wkb})
        _geom_row = conn.execute(
            sa.text("SELECT id FROM geometries WHERE geom_hash = :h"),
            {"h": geom_hash},
        ).fetchone()
        assert _geom_row is not None
        geom_id: int = _geom_row[0]

        # --- request ---
        conn.execute(
            sa.text(ignore_req),
            {
                "gid": geom_id,
                "sat": satellite.shortName,
                "ph": params_hash,
                "red": json.dumps(sorted(reducers)) if reducers else None,
                "sub": subsampling_max_pixels,
                "sd": start_date,
                "ed": end_date,
                "now": datetime.now(UTC).isoformat(),
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

        # Skip if rows already exist for this request
        already_exists = conn.execute(
            sa.text(f'SELECT 1 FROM "{table_name}" WHERE request_id = :rid LIMIT 1'),
            {"rid": req_id},
        ).fetchone()
        if already_exists:
            return

        # --- satellite rows ---
        present_band_cols = [c for c in band_cols if c in df.columns]
        col_names_sql = ", ".join(['"request_id"', '"timestamp"', *[f'"{c}"' for c in present_band_cols]])
        placeholders_sql = ", ".join([
            ":request_id",
            ":timestamp",
            *[f":band_{i}" for i in range(len(present_band_cols))],
        ])

        ts_col = "timestamp" if "timestamp" in df.columns else None
        rows_to_insert = []
        for _, row in df.iterrows():
            record: dict = {
                "request_id": req_id,
                "timestamp": str(row[ts_col]) if ts_col and pd.notna(row[ts_col]) else None,
            }
            for i, c in enumerate(present_band_cols):
                record[f"band_{i}"] = None if pd.isna(row[c]) else float(row[c])
            rows_to_insert.append(record)

        if rows_to_insert:
            conn.execute(
                sa.text(f'INSERT INTO "{table_name}" ({col_names_sql}) VALUES ({placeholders_sql})'),
                rows_to_insert,
            )


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------


def print_cache_status(engine: sa.Engine | None = None) -> None:
    """Print a summary of the current cache state to stdout.

    Shows the total number of cached images and SITS, then a per-sensor
    breakdown (row count in each satellite table).
    If *engine* is None the module-level engine is used; if no engine is
    available the function prints a "cache not initialised" notice and returns.
    """
    eng = engine or _cache_engine
    if eng is None:
        print("[AgriGEE cache] Cache not initialised.")
        return

    is_pg = eng.dialect.name == "postgresql"

    _SL_SYSTEM_TABLES = {
        "geometries", "requests", "spatial_ref_sys", "geometry_columns",
        "views_geometry_columns", "virts_geometry_columns", "spatialite_history",
        "sql_statements_log", "geometry_columns_statistics",
        "geometry_columns_field_infos", "geometry_columns_time",
        "geometry_columns_auth", "data_licenses", "KNN", "KNN2",
        "ElementaryGeometries",
    }
    _PG_SYSTEM_TABLES = {"geometries", "requests", "spatial_ref_sys", "geometry_columns"}

    with eng.connect() as conn:
        _count_row = conn.execute(sa.text("SELECT COUNT(*) FROM requests")).fetchone()
        assert _count_row is not None
        total_sits: int = _count_row[0]

        if is_pg:
            rows = conn.execute(sa.text(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
            )).fetchall()
            sat_tables = sorted(r[0] for r in rows if r[0] not in _PG_SYSTEM_TABLES)
        else:
            rows = conn.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table'")).fetchall()
            sat_tables = sorted(r[0] for r in rows if r[0] not in _SL_SYSTEM_TABLES)

        images_by_sensor: dict[str, int] = {}
        for tbl in sat_tables:
            _tbl_row = conn.execute(sa.text(f'SELECT COUNT(*) FROM "{tbl}"')).fetchone()
            assert _tbl_row is not None
            images_by_sensor[tbl] = _tbl_row[0]

    total_images = sum(images_by_sensor.values())
    backend = f"PostGIS ({_PG_DB_NAME})" if is_pg else f"SpatiaLite ({DEFAULT_DB_PATH})"

    lines = [
        f"[AgriGEE cache] Status  [{backend}]",
        f"  Total images : {total_images}",
        f"  Total SITS   : {total_sits}",
        "  Images per sensor:",
    ]
    for sensor, count in images_by_sensor.items():
        lines.append(f"    {sensor:<30} {count:>7}")

    print("\n".join(lines))


def init_cache(db_path: pathlib.Path = DEFAULT_DB_PATH) -> sa.Engine:
    """Initialise the SITS cache database (SpatiaLite or PostGIS).

    If the ``AGRIGEE_PG_HOST``, ``AGRIGEE_PG_USER``, and
    ``AGRIGEE_PG_PASSWORD`` environment variables are set, a PostGIS database
    named ``agrigeelite`` is used (created automatically if missing).
    Otherwise a local SpatiaLite file is used.

    All schema operations are idempotent — safe to call on every library
    start-up.

    Returns
    -------
    sqlalchemy.Engine
        Engine bound to the initialised database.
    """
    global _cache_engine

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
        Sentinel2(),  # s2sr
        Sentinel2(use_sr=False),  # s2
        Sentinel1GRD(),  # s1a
        Sentinel1GRD(ascending=False),  # s1d
        Landsat5(),  # l5sr
        Landsat7(),  # l7sr
        Landsat8(),  # l8sr
        Landsat9(),  # l9sr
        ModisDaily(),  # modis
        Modis8Days(),  # modis8days
        NAIP(),  # naip
        HLSSentinel2(),  # hls_s2
        HLSLandsat(),  # hls_l8
        PALSAR2ScanSAR(),  # palsar2
        CopernicusDEM(),  # copdem
        ANADEM(),  # anadem
        WRBSoilClasses(),  # wrb_soil_classes
        MapBiomas(),  # mapbiomasmajclass
        SatelliteEmbedding(),  # satembed
    ]

    if _pg_env_set():
        logger.info("AgriGEE cache: using PostGIS backend (host=%s)", os.environ["AGRIGEE_PG_HOST"])
        engine = _make_pg_engine()
        with engine.begin() as conn:
            _ensure_postgis_extension(conn)
            _ensure_geometries_table_pg(conn)
            _ensure_requests_table_pg(conn)
            for sat in satellites:
                _ensure_satellite_table_pg(conn, sat.shortName, _get_band_columns(sat))
    else:
        engine = _make_spatialite_engine(db_path)
        with engine.begin() as conn:
            _ensure_spatial_metadata(conn)
            _ensure_geometries_table_sl(conn)
            _ensure_requests_table_sl(conn)
            for sat in satellites:
                _ensure_satellite_table_sl(conn, sat.shortName, _get_band_columns(sat))

    _cache_engine = engine
    return engine


# ---------------------------------------------------------------------------
# Cache clearing
# ---------------------------------------------------------------------------


def clear_cache(
    sits_db: bool = True,
    sits_files: bool = True,
    image_files: bool = True,
) -> None:
    """Delete all locally cached data produced by AgriGEE.lite.

    Parameters
    ----------
    sits_db : bool, default True
        Drop the SpatiaLite / PostGIS SITS cache database.
        For SpatiaLite this deletes ``~/.cache/agrigee_lite/sits_cache.db``.
        For PostGIS the ``agrigeelite`` database tables are truncated and
        the schema is dropped.
    sits_files : bool, default True
        Remove all downloaded SITS CSV chunks under
        ``~/.cache/agrigee_lite/sits/``.
    image_files : bool, default True
        Remove all downloaded image ZIP files under
        ``~/.cache/agrigee_lite/images/``.
    """
    global _cache_engine

    cache_root = pathlib.Path.home() / ".cache" / "agrigee_lite"
    removed: list[str] = []

    # --- SITS database ---
    if sits_db:
        eng = _cache_engine
        if eng is not None and eng.dialect.name == "postgresql":
            # PostGIS: drop all satellite tables + core tables inside the DB.
            with eng.begin() as conn:
                conn.execute(sa.text("DROP SCHEMA public CASCADE"))
                conn.execute(sa.text("CREATE SCHEMA public"))
            removed.append("PostGIS schema (agrigeelite)")
        else:
            # SpatiaLite: close the engine, delete the file, then recreate.
            if eng is not None:
                eng.dispose()
                _cache_engine = None
            db_file = DEFAULT_DB_PATH
            if db_file.exists():
                db_file.unlink()
                removed.append(str(db_file))
            init_cache(db_file)

    # --- Downloaded SITS CSV chunks ---
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

    # --- Downloaded image ZIPs ---
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
