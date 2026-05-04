# GeoPolars Migration Plan

## Goal

Make the library use `geopolars` as the internal geospatial dataframe backend, while replacing `topojson`-based simplification with GeoPolars-native simplification.

Also migrate all outputs that currently return `pandas.DataFrame` so they return `polars.DataFrame` instead.

Input compatibility may remain transitional, but output types should move to Polars early because this is a performance goal, not just an internal refactor.

## Temporary decision

For now, do not implement simplification in the GeoPolars path.

Current repo state:

- `pixi.toml` already includes `geopolars==0.1.0a4`
- local inspection showed no obvious public `GeoSeries.simplify()` method in that version

So the first GeoPolars migration should keep geometries unsimplified and leave an explicit `TODO` exactly where simplification should later be added.

## Scope

Primary files:

- `agrigee_lite/misc.py`
- `agrigee_lite/get/sits.py`
- `agrigee_lite/cache/backend.py`
- `agrigee_lite/ee_utils.py`
- `pyproject.toml`
- `pixi.toml`
- docs that mention GeoPandas/TopoJSON

## Execution plan

### Step 0. Convert dataframe outputs from Pandas to Polars

Do this first, because later geo and cache refactors should target the final output contract.

Primary targets:

- `agrigee_lite/get/sits.py`
- cache read paths that currently materialize `pandas.DataFrame`
- any helper that still assumes Pandas as the final tabular result

Required changes:

- make SITS-returning functions return `polars.DataFrame`
- minimize `to_pandas()` usage in the download and cache paths
- keep Pandas only where a dependency strictly requires it
- update type hints, docstrings, and tests to reflect Polars outputs

If a public function currently returns `pandas.DataFrame`, treat changing it to `polars.DataFrame` as an intentional API change, and propagate that change consistently.

### Step 1. Add a geo compatibility layer

Create a small internal module, for example `agrigee_lite/_geo_compat.py`.

It should:

- accept `geopandas.GeoDataFrame | geopolars.GeoDataFrame`
- normalize to `geopolars.GeoDataFrame`
- provide conversion helpers back to `GeoPandas` only where still needed
- isolate backend-specific operations such as:
  - getting the geometry series
  - iterating shapely geometry objects
  - converting to GeoJSON-ready objects
  - hashing geometry rows safely

Do not spread ad-hoc conversions across the codebase.

### Step 2. Migrate `misc.py` first

Refactor `agrigee_lite/misc.py` so that:

- `simplify_gdf()` operates on `geopolars`
- `h3_clustering()` operates on `geopolars`
- `create_gdf_hash()` accepts the normalized internal geo frame type

Remove the `topojson` dependency from this path, but do not replace it with active simplification yet.

Expected interim behavior:

- keep `h3_coarse`, `h3_fine`, and `cluster_id`
- skip all geometry simplification
- leave geometry unchanged after clustering
- add an explicit `TODO` in the simplification slot, for example inside `simplify_gdf()` or the cluster-processing branch

Recommended placeholder intent:

- `TODO: reintroduce geometry simplification here once GeoPolars exposes a stable native simplify API`

### Step 3. Migrate `get/sits.py` to normalized GeoPolars input

Update `sanitize_and_prepare_input_gdf()` to:

- accept `GeoPandas` or `GeoPolars`
- validate the input schema
- normalize to `geopolars`
- return `geopolars`

Recommended approach:

- keep `pandera` validation at the public-input boundary if it still works best there
- convert to `geopolars` immediately after validation

Then update the async batch pipeline to use the normalized internal type throughout:

- chunk creation
- cluster filtering
- cache coverage lookup inputs
- queue payload types
- final result assembly as `polars.DataFrame`

Minimize back-conversion to pandas/geopandas.

### Step 4. Adapt cache lookup to backend-neutral geometry access

Refactor `agrigee_lite/cache/backend.py` so batch coverage logic does not assume `GeoPandas` APIs.

Target changes:

- accept normalized geo frame input
- compute `h3_coarse` and `h3_fine` from normalized columns
- replace `GeoPandas`-specific `.apply(...)`, `.loc`, `.at`, `.geometry.geom_type` usage with backend-neutral helpers or explicit iteration

Keep cache semantics unchanged:

- same geometry hash behavior
- same gap computation behavior
- same point fast path

Also push cache fetch results toward `polars.DataFrame` so the calling pipeline does not bounce through Pandas unnecessarily.

### Step 5. Remove GeoPandas file-export dependency from EE conversion

Refactor `ee_gdf_to_feature_collection()` in `agrigee_lite/ee_utils.py`.

Current issue:

- it depends on `GeoPandas.to_crs()` and `to_file(..., driver="GeoJSON")`

Target:

- build the GeoJSON-like feature collection directly from normalized geometries and properties
- only use `GeoPandas` as a temporary fallback if absolutely necessary

This reduces one of the last reasons to keep GeoPandas in the runtime hot path.

### Step 6. Clean dependencies and docs

After code migration is complete:

- remove `topojson` from `pyproject.toml`
- remove `topojson` from `pixi.toml`
- decide whether `geopandas` remains required or becomes transitional/optional
- update docs and examples to describe:
  - GeoPolars as the internal engine
  - GeoPandas as compatibility input during transition, if still true

## Validation checklist

Add or update tests for:

- output type migration from Pandas to Polars
- `simplify_gdf()` on `Point`, `Polygon`, `MultiPolygon`
- duplicate geometry simplification behavior
- `h3_clustering()` ordering and `cluster_id`
- `sanitize_and_prepare_input_gdf()` with GeoPandas input
- `sanitize_and_prepare_input_gdf()` with GeoPolars input
- cache batch coverage with normalized geo frames
- `ee_gdf_to_feature_collection()` output equivalence

## Acceptance criteria

The task is complete when all of the following are true:

- functions that previously returned `pandas.DataFrame` now return `polars.DataFrame`
- internal batch geospatial flow uses `geopolars`
- `topojson` is no longer used
- public APIs still accept existing GeoPandas inputs unless intentionally changed
- current no-simplification behavior is tested and documented
- `pixi run pyright` returns `0 errors, 0 warnings`

## Recommended execution order

1. Convert DataFrame outputs from Pandas to Polars.
2. Add `_geo_compat.py`.
3. Refactor `misc.py` with the no-simplification GeoPolars path and an explicit `TODO`.
4. Refactor `get/sits.py`.
5. Refactor `cache/backend.py`.
6. Refactor `ee_utils.py`.
7. Remove `topojson` and update docs.

## Notes for the implementing LLM

- Do not introduce `.pyi` files.
- Keep inline typing only.
- Follow existing AGENTS.md Pyright rules.
- Prefer incremental commits and keep the public API stable unless a code-level blocker forces a change.
- Do not block the migration on simplification support.
- Remove TopoJSON usage now, keep geometries unsimplified, and leave a clear `TODO` at the future simplification hook.
