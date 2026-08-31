## AGENTS (compact LLM-oriented) — agrigee_lite_client

Subproject inside the AgriGEE.lite monorepo. See `../AGENTS.md` for the
server; this file covers only the HTTP client. Read `SPECS.md` in full
before touching this code — the design decisions (naive on purpose, no
install extras, geopandas optional-only) are already made and justified
there; don't reopen them without a new reason.

Run checks:

```bash
cd agrigee_lite_client
uv pip install -e ".[dev]"
pytest
ruff check .
```

## Key rules (short)

- `geopandas` is never a hard dependency, but `multiple_sits()` accepts it as
  input via lazy detection (`_geo_compat.py`: `try: import geopandas` inside
  the function, never at module top-level). A top-level `import geopandas`
  anywhere in the package is a bug — callers who only use `sits()`/geopolars/
  polars must never pay that import. `geopandas` as a **test-only**
  dependency (`test_geoparquet.py`, `test_geo_compat.py`) is fine.
- Fixed dependencies, no extras: `geopolars`, `shapely`, `httpx`. Do not add
  `[async]`, `[shapely]`, or any optional install variant — explicit user
  decision, don't reintroduce.
- Version always matches the server's (`../pyproject.toml`). Every server
  version bump gets the same bump here, same commit.
  `tests/test_version_matches_server.py` breaks CI on drift.
- Sync and async are deliberately duplicated code (`_client.py` /
  `_async_client.py`) — do not build a sync/async unification layer (e.g.
  `unasync`) on top. `_get.py` already holds everything that's genuinely
  shared (payload building, response parsing); only I/O (`httpx` calls,
  `sleep`/`asyncio.sleep`) is duplicated.
- `get.sits()` is sugar over `get.multiple_sits()`, not a separate endpoint —
  don't give it its own HTTP path without reviewing SPECS.md §6 and §11
  first (a synchronous `/sits/single/file` server endpoint is explicitly
  deferred future work, don't implement it preemptively).
- GeoParquet metadata is hand-built (`_geoparquet.py`) via
  `pyarrow.Table.replace_schema_metadata`, because `geopolars` 0.1.0-alpha.4
  doesn't write the `"geo"` key on its own (`GeoDataFrame.write_parquet` is
  the plain `polars.DataFrame.write_parquet`). Reconfirm that behavior on
  the installed `geopolars` version before changing this approach — it may
  have changed.
- CRS is always `EPSG:4326`/`OGC:CRS84` — the client never reprojects. Don't
  add `pyproj`/reprojection without discussing first (breaks the minimal-
  dependency promise). A `geopandas.GeoDataFrame` with a different CRS is
  rejected with `ValueError`, not reprojected.
- Don't add configurable retry/backoff, local caching, authentication, a
  CLI, or image decoding (`numpy`/`tifffile`/`rasterio`) without updating
  SPECS.md §10 first — explicit non-goals of this v1, not oversights.
- **`pip install agrigee_lite_client` fails on any pip ≥24.1** — not our
  bug: `geopolars==0.1.0a4` on PyPI ships a non-PEP-440 dependency
  specifier (`pyarrow>=4.0.*`), which recent pip refuses outright. `uv pip
  install` tolerates it and works; so does `pip<24.1`. Same caveat applies
  to `agrigee_lite` itself. Nothing to fix here short of a corrected
  `geopolars` release — documented in both READMEs, don't waste time
  re-diagnosing it.
- **No new server endpoint just to support this client.** One was proposed
  (`POST /satellites/date_range`) and rejected — the user didn't want the
  server growing surface area purely for the client's benefit. `GET
  /version` is the one exception, added specifically because `/health`
  already has consumers and must not change shape. Prefer extending an
  existing endpoint over adding a new one; if that's not possible, ask
  before adding one.
- `_satellite_dates.py` is generated, not hand-edited. Regenerate with:

```bash
python scripts/generate_satellite_dates.py > agrigee_lite_client/_satellite_dates.py
```

  (needs `agrigee_lite` importable — use the `all-features` pixi env)
  whenever a satellite is added/removed or its range changes server-side,
  and bump both packages' version together (SPECS.md §2.1).
  `test_satellite_dates_match_server.py` catches drift automatically
  whenever both packages are installed side by side, but doesn't replace
  regenerating after a real server-side change.
- `pl.col(...).cast(pl.Datetime, strict=False)` silently nulls a `Utf8`
  column instead of parsing it, as of polars 1.43 — it used to parse; that
  behavior is deprecated/gone, `str.to_datetime(strict=False)` is the
  replacement. Bit this code once (every row got silently dropped).
  `_validation.py::_as_datetime_expr` checks dtype and picks the right one —
  don't copy the server's `sanitize_and_prepare_input_gdf` pattern
  (`agrigee_lite/get/sits.py`) verbatim; it only works there because the
  column already arrives converted via `pandas.to_datetime`.

## Tests

- `test_geoparquet.py` — no network, validates bytes against
  `geopandas.read_parquet`.
- `test_validation.py`, `test_geo_compat.py`, `test_satellite_dates.py` — no
  network, test `_validation.py`/`_geo_compat.py`/`_satellite_dates.py` in
  isolation.
- `test_satellite_dates_match_server.py` — only runs when `agrigee_lite` is
  importable (`pytest.importorskip`); this is what actually catches drift
  between the static table and the server's `REGISTRY`.
- `test_client_mocked.py` / `test_async_client_mocked.py` — no network, use
  `httpx.MockTransport` + `tests/_fake_server.py` (a stateful fake of the
  real API: health, version, satellites, job lifecycle). Add new scenarios
  here, not against a real server.
- `test_client_against_live_api.py` — only runs with `AGRIGEE_TEST_API_URL`
  set; not part of the default test run (depends on network + the server's
  Earth Engine credentials).
- `test_version_matches_server.py` — always runs, the version lock.

## Docker Hub

This subproject has no image of its own — it's consumed by whoever runs
`uv pip install agrigee_lite_client`. The relevant Docker image is the
server's (`mateuspinto/agrigee-lite`, see `../AGENTS.md`).
