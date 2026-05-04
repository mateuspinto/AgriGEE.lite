# AGENTS.md — Instructions for AI Agents Working on This Repository

## Type Checking

This project uses **Pyright** (via `pixi`) as its type checker. Pyright is configured in `pyproject.toml` under `[tool.pyright]`.

### Running Pyright

```bash
pixi run pyright
```

**The target is always 0 errors, 0 warnings.** Do not leave the session until `pixi run pyright` exits cleanly.

### Key Configuration Decisions

| Setting | Value | Reason |
|---|---|---|
| `typeCheckingMode` | `standard` | EE SDK objects are highly dynamic; strict mode generates too many false-positives |
| `reportUnknownVariableType` / `MemberType` / `ArgumentType` | `false` | Earth Engine's Python SDK resolves most bindings at runtime |
| `reportMissingTypeStubs` | `false` | Many scientific libraries ship without stubs |
| `reportUnnecessaryTypeIgnoreComment` | `true` | Stale `# type: ignore` comments must be removed |
| `exclude` includes `agrigee_lite/api` | intentional | `fastapi` and `uvicorn` are optional extras not installed in the base pixi environment |

### Inline Annotations Only — No `.pyi` Files

This package uses **inline type annotations** in `.py` files exclusively. It ships `agrigee_lite/py.typed` (PEP 561 marker). There are no `.pyi` stub files. Do not create `.pyi` files — Pyright ignores `.py` when a `.pyi` exists for the same module, which causes annotation drift.

### Recurring Patterns and How to Fix Them

**1. Earth Engine arithmetic with Python literals**

EE type stubs declare `.add()`, `.divide()`, `.lt()`, `.neq()`, etc. as accepting only `ee.Image` or `ee.Number`, not bare Python `int`/`float`. Always wrap literals:

```python
# Wrong — Pyright error: float not assignable to Integer
img.divide(10000)
img.add(100).divide(16_100)
img.lt(-30.0)

# Correct
img.divide(ee.Number(10000))
img.add(ee.Number(100)).divide(ee.Number(16_100))
img.lt(ee.Number(-30.0))
```

**2. Parameter reassignment confuses type narrowing**

When a parameter is `set[str] | None`, reassigning it to a `list[str]` narrows inconsistently. Use a local variable with a trailing underscore instead:

```python
# Wrong — Pyright loses track of the narrowed type
def __init__(self, bands: set[str] | None = None):
    bands = sorted({"red", "nir"}) if bands is None else sorted(bands)

# Correct
def __init__(self, bands: set[str] | None = None):
    bands_: list[str] = sorted({"red", "nir"}) if bands is None else sorted(bands)
```

**3. `selectedIndices` base class type**

`AbstractSatellite.selectedIndices` is `list[tuple[str, str, str]]`. All satellite subclasses assign 3-tuples `(expression, name, numeral_name)`. Do not re-declare it as `list[str]` in subclasses.

**4. `getInfo()` returns `Any | None`**

EE's `.getInfo()` return type is `Any | None`. When you need a concrete type, use `cast`:

```python
from typing import cast
names: list[str] = cast(list[str], collection.aggregate_array("band").getInfo())
```

**5. `fetchone()` is subscriptable only after a None check**

SQLAlchemy's `fetchone()` returns `Row | None`. Always assert before subscripting:

```python
row = conn.execute(sa.text("SELECT id FROM ..."), {...}).fetchone()
assert row is not None
value: int = row[0]
```

**6. Optional-dependency imports**

Imports guarded by `try/except ImportError` (e.g., `matplotlib`, `smart_open`) should carry a `# pyright: ignore[reportMissingImports]` comment on the import line. If an entire module depends on an optional extra that is not installed in the base pixi environment, exclude the module path in `pyproject.toml` under `[tool.pyright] exclude`.

**7. `reduceRegion(maxPixels=...)` requires `int`**

EE's `reduceRegion` expects `Integer | None` for `maxPixels`. The parameter `subsampling_max_pixels` is typed as `float`. Always cast:

```python
.reduceRegion(maxPixels=int(subsampling_max_pixels), ...)
```

**8. `hasattr` guards do not narrow attribute access**

Pyright does not narrow `satellite.classes` to a known type after `hasattr(satellite, "classes")`. Use `getattr` instead:

```python
if hasattr(satellite, "classes"):
    values = getattr(satellite, "classes").values()
```

**9. `asyncio.gather` with `return_exceptions=True`**

When iterating results, use `isinstance(res, BaseException)` (not `Exception`) to narrow away errors, because `asyncio.gather` can surface `BaseException` subclasses:

```python
for cid, res in zip(batch, results):
    if isinstance(res, BaseException):
        handle_error(cid)
    else:
        handle_success(res)  # res is now narrowed
```

### Adding a New Satellite Class

1. Inherit from `OpticalSatellite`, `RadarSatellite`, or `SingleImageSatellite`.
2. Use `bands_: list[str]` and `indices_: list[str]` locals (not reassigning the parameters).
3. Do not annotate `self.selectedIndices` in the subclass — it inherits `list[tuple[str, str, str]]` from the base.
4. Wrap all EE arithmetic literals with `ee.Number(...)`.
5. Cast `maxPixels=int(subsampling_max_pixels)` in every `reduceRegion` call.
6. Run `pixi run pyright` and confirm 0 errors before finishing.

---

## Library Overview

GEE download library. Two data planes: **SITS** (tabular time series → `pd.DataFrame`) and **Images** (GeoTIFF ZIPs → `list[str]` of dates).

### Entry point

```python
import agrigee_lite as agl
agl.initialize()
# = _install_uvloop() + init_cache() + print_cache_status() + ee_quick_start() + _tune_ee_http()
```

---

## Satellite classes — `agrigee_lite.sat`

### Hierarchy

```
AbstractSatellite
├── OpticalSatellite     cloud-affected, reflectance 0–1
│   ├── Sentinel2        shortName: s2sr (SR) / s2 (TOA)
│   ├── Landsat5/7/8/9
│   ├── HLSLandsat, HLSSentinel2
│   ├── NAIP
│   └── TwoSatelliteFusion   fuses two OpticalSatellites by common dates
├── RadarSatellite       SAR, cloud-free, dB
│   ├── Sentinel1GRD
│   └── PALSAR2ScanSAR
├── DataSourceSatellite  derived products
│   ├── MapBiomas, Modis8Days, ModisDaily, SatelliteEmbedding
└── SingleImageSatellite no time axis, exposes .image() not .imageCollection()
    ├── ANADEM, CopernicusDEM, WRBSoilClasses
```

### Key attributes on every satellite

| Attr | Type | Notes |
|---|---|---|
| `shortName` | str | cache table name + column prefix |
| `startDate` / `endDate` | str ISO | sensor valid range |
| `availableBands` | dict[str,str] | friendly→GEE band name |
| `selectedBands` | list[(str,str)] | (friendly, numeral_output_col) |
| `selectedIndices` | list[(expr,name,col)] | spectral indices |
| `pixelSize` | int | metres |
| `toDownloadSelectors` | list[str] | output column names |
| `log_dict()` | dict | serialisable config used for cache key |
| `availableIndices` | property dict | indices computable from selectedBands |

### Construction

```python
s2   = agl.sat.Sentinel2(bands={"red","nir"}, indices={"ndvi"}, use_sr=True)
l8   = agl.sat.Landsat8(bands={"red","nir","swir1"})
s1   = agl.sat.Sentinel1GRD()
dem  = agl.sat.CopernicusDEM()
fuse = agl.sat.TwoSatelliteFusion(l8, s2)   # bands prefixed 8_ and 7_
```

---

## Download functions — `agrigee_lite.get`

### SITS — single geometry

```python
df = agl.get.sits(
    geometry,                      # shapely Polygon | MultiPolygon | Point, WGS-84
    start_date, end_date,          # str or pd.Timestamp
    satellite,
    reducers=None,                 # set[str] | None; None = per-pixel
    subsampling_max_pixels=1_000,  # >1 absolute px; ≤1 fraction of area
)  # -> pd.DataFrame  cols: timestamp + band/index cols
```

Cache-aware: only downloads gaps vs. already-cached intervals.

### SITS — GeoDataFrame batch (async)

```python
df = await agl.get.async_multiple_sits(
    gdf,                             # gpd.GeoDataFrame with geometry, start_date, end_date
    satellite,
    reducers=None,
    original_index_column_name="original_index",
    start_date_column_name="start_date",
    end_date_column_name="end_date",
    subsampling_max_pixels=1_000,
    chunksize=10,                    # geometries per GEE HTTP request
    max_parallel_downloads=40,
    max_retries_per_chunk=8,
    force_redownload=False,
)  # -> pd.DataFrame  includes original_index_column_name column
```

**Flow inside:**
1. `sanitize_and_prepare_input_gdf` — validate schema, clip to satellite range, H3 sort + TopoJSON simplify per cluster
2. `fetch_sits_batch_coverage` — 3-stage H3→hash→SQL batch lookup
3. Full cache hits → `_finalize_from_cache`; partial/miss → `uncached_request_rows`
4. `build_ee_expression` → `asyncio.gather` + AIMD semaphore → `fetch_with_retry` per chunk
5. Concat `pl.DataFrame` results → `prepare_output_df` → groupby orig_idx → `store_sits`

Valid reducers: `min max mean median mode std var kurt skew p<N>` (e.g. `p25 p75`).
Multiple reducers → columns named `<band>_<reducer>`.

### SITS — GDrive / GCS batch

`get.multiple_sits_gdrive`, `get.multiple_sits_gcs` — use GEE Export tasks for very large requests.

### Images

```python
# Async (preferred)
dates = await agl.get.async_images(
    geometry,             # Polygon | MultiPolygon
    start_date, end_date,
    satellite,
    invalid_images_threshold=0.5,   # min valid-pixel fraction to keep image
    max_parallel_downloads=40,
    force_redownload=False,
    image_indices=None,             # list[int] | None — download subset by position
    max_retries_per_chunk=5,
)  # -> list[str]  YYYY-MM-DD dates of downloaded images

# Sync wrapper
dates = agl.get.images(...)
```

Files at `~/.cache/agrigee_lite/images/<hash>/<date>.zip`.
Cache: existing `*.zip` stems = hits. `force_redownload=True` deletes ZIPs first.
Cache key: `create_dict_hash(metadata_dict)` including centroid, satellite config, dates.

---

## Cache — `agrigee_lite.cache`

### Backend selection

| Condition | Backend |
|---|---|
| default | DuckDB at `~/.cache/agrigee_lite/sits_cache.duckdb` |
| `AGRIGEE_PG_HOST` + `AGRIGEE_PG_USER` + `AGRIGEE_PG_PASSWORD` set | PostGIS db=`agrigeelite` |

`CacheEngine = duckdb.DuckDBPyConnection | sa.Engine`

### Schema

**`geometries`** — one row per unique shape
- `id PK`, `geom_hash TEXT UNIQUE` (SHA-1 WKB), `geometry BLOB`, `repr_point_x/y DOUBLE`, `geom_type TEXT`, `h3_coarse TEXT` (res 5), `h3_fine TEXT` (res 8)
- Indexes on `geom_hash`, `h3_coarse`, `h3_fine`, `(repr_point_x, repr_point_y)`

**`sits_jobs`** — one row per (geometry, satellite, params, date range) fetch
- `id PK`, `job_hash TEXT UNIQUE`, `geometry_id FK`, `satellite_short_name`, `params_hash`, `reducers`, `subsampling_max_pixels`, `start_date`, `end_date`, `fetched_at`
- Unique constraint on `(geometry_id, satellite_short_name, params_hash, start_date, end_date)`

**`<satellite_shortname>`** — one table per satellite
- `id PK`, `job_id FK→sits_jobs`, `timestamp TIMESTAMPTZ`, `<band> DOUBLE` per band/index

### Cache identity

```
geom_hash   = SHA-1(geometry.wkb)
params_hash = SHA-1({"satellite": sat.log_dict(), "reducers": sorted_or_null, "subsampling_max_pixels": ...})
job_hash    = SHA-1(f"{geom_hash}|{params_hash}|{start_date}|{end_date}")
```

### Public API

```python
from agrigee_lite.cache import init_cache, clear_cache, print_cache_status
from agrigee_lite.cache.backend import (
    get_engine,
    fetch_sits_with_gaps,       # (engine, geom, start, end, sat, reducers, sub) -> (df, gaps)
    fetch_sits_batch_coverage,  # (engine, gdf, sat, ...) -> dict[int, (job_ids, gaps)]
    fetch_sits_by_job_ids,      # (engine, job_ids, sat, start, end) -> dict[job_id, df]
    store_sits,                 # (engine, pd.DataFrame, geom, start, end, sat, reducers, sub)
    store_sits_polars,          # same signature but df is pl.DataFrame; DuckDB uses Arrow path
)

init_cache(satellites=[...], db_path=DEFAULT_DB_PATH)   # idempotent; called by initialize()
clear_cache(sits_db=True)       # drop DuckDB file or reset PostGIS public schema
clear_cache(image_files=True)   # delete ~/.cache/agrigee_lite/images/
print_cache_status()            # logs sits_jobs count, geometries count, satellite tables
```

### Partial date reuse

Requesting 2023–2025 when 2022–2024 is cached → only 2025 downloaded.
Gaps computed by `_compute_gaps(query_start, query_end, covered_intervals)` using `datetime.date` arithmetic.
Gaps tracked at `sits_jobs` row level — cloudy periods with no observations are not re-fetched.

### Batch lookup — 3-stage pipeline (non-point geometries)

1. SQL scan `geometries WHERE h3_coarse IN (all_input_coarse_cells)` → candidate set
2. Keep candidates whose `h3_fine` intersects input fine-cell set
3. SHA-1 computed only for H3-fine survivors; match against candidate `geom_hash` → `geometry_id`
4. Single SQL on `sits_jobs` for `(geometry_id, params_hash, date_overlap)`

Point fast path: `SELECT id FROM geometries WHERE repr_point_x = ? AND repr_point_y = ?` — no SHA-1.

### Native Polars→DuckDB write

`store_sits_polars` / `_store_sits_duck_polars`: Arrow registration instead of `executemany`:

```python
obs_pl = pl_df.select(cols).with_columns(pl.lit(job_id).alias("job_id"))
conn.register("_obs_tmp", obs_pl.to_arrow())
conn.execute(f'INSERT INTO "{table_name}" ({col_list}) SELECT {col_list} FROM _obs_tmp')
conn.unregister("_obs_tmp")
```

PostGIS path always falls back to pandas via `_store_sits_pg`.

---

## Utilities — `agrigee_lite.misc`

| Function | Purpose |
|---|---|
| `h3_clustering(gdf, coarse=5, fine=8)` | sort GDF by H3 cells, add `cluster_id`, simplify via TopoJSON per cluster |
| `simplify_gdf(gdf, tol=0.001)` | TopoJSON simplification preserving shared edges, deduplicates geometries |
| `random_points_from_gdf(gdf, n=10, buffer=-10)` | N random grid points per geometry |
| `get_sample_gdf()` | load bundled `data/sample.parquet` |
| `create_gdf_hash(gdf, start_col, end_col)` | SHA-1 of centroid coords + dates |
| `create_dict_hash(d)` | SHA-1 of JSON dict (sets → sorted lists) |

---

## Configuration — env vars

| Var | Default | Notes |
|---|---|---|
| `AGRIGEE_MAX_PARALLEL_DOWNLOADS` | 40 | AIMD ceiling |
| `AGRIGEE_MAX_URL_WORKERS` | 10 | ThreadPool for GEE URL generation |
| `AGRIGEE_MAX_RETRIES_PER_CHUNK` | 8 | retries before chunk gives up |
| `AGRIGEE_AIOHTTP_TIMEOUT_SECONDS` | 600 | per-request timeout |
| `AGRIGEE_SITS_CHUNKSIZE` | 10 | geometries per GEE HTTP request |
| `AGRIGEE_AIMD_SUCCESS_STRIDE` | 5 | successes before concurrency +1 |
| `AGRIGEE_AIMD_INITIAL_DOWNLOADS` | =MAX | starting concurrency |
| `AGRIGEE_USE_UVLOOP` | true | |
| `AGRIGEE_PG_HOST/USER/PASSWORD` | — | enable PostGIS |
| `AGRIGEE_PG_PORT` | 5432 | |

---

## REST API — `agrigee_lite.api`

Optional extra (`pip install agrigee_lite[api]`). FastAPI + uvicorn.

```bash
agl_api [--host 127.0.0.1] [--port 8000] [--reload]
```

| Method | Path | Behaviour |
|---|---|---|
| GET | `/health` | `{"status":"ok"}` |
| GET | `/satellites` | list satellite names |
| POST | `/sits/single` | sync (thread pool), returns records JSON array |
| POST | `/sits/multiple` | async, returns 202 + `job_id` |
| POST | `/images` | async, returns 202 + `job_id` |
| GET | `/jobs/{job_id}` | poll: pending→running→completed/failed + result |

`JobStore` is in-memory, process-local, lost on restart.
Satellite JSON: `{"name": "Sentinel2", "params": {"bands": ["red","nir"]}}`.

---

## File map

```
agrigee_lite/
├── __init__.py              initialize(), re-exports get/sat/vis/cache
├── config.py                all env-var constants
├── ee_utils.py              GEE auth, HTTP tuning, EE helpers
├── misc.py                  h3_clustering, simplify_gdf, hash utils
├── vegetation_indices.py    VEGETATION_INDICES dict
├── sat/
│   ├── abstract_satellite.py   AbstractSatellite + OpticalSatellite/Radar/DataSource/SingleImage
│   ├── sentinel2.py, landsat.py, sentinel1.py, ...   one file per sensor
│   └── unified_satellite.py    TwoSatelliteFusion
├── get/
│   ├── sits.py              download_single_sits, download_multiple_sits_async, chunks_gdrive/gcs
│   └── image.py             download_single_image, download_multiple_images_async
├── cache/
│   ├── backend.py           DuckDB+PostGIS schema, read/write, gap logic, batch lookup
│   └── __init__.py          re-exports init_cache, clear_cache, print_cache_status
└── api/
    ├── __init__.py          create_app(), serve()
    ├── _satellites.py       REGISTRY: name→constructor
    ├── _jobs.py             JobStore in-memory
    ├── _models.py           Pydantic request/response models
    └── routes/              sits.py, images.py, jobs.py
```
