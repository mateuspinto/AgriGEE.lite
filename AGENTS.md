## AGENTS (compact LLM-oriented)

Purpose: short, machine-friendly guidance for automated agents and maintainers to make predictable, safe edits.

- Run checks: `pixi run pyright` (expect 0 errors/warnings) and `pixi run pytest` in CI.
- Style: keep changes minimal and focused; preserve public APIs and typing.
- Tests: add unit tests for behaviour-first changes; avoid changing many files in one PR.

Key rules (short):

- Always prefer explicit types and local temporaries (e.g. `bands_`) to avoid Pyright narrowing issues.
- Wrap Earth Engine numeric literals in `ee.Number(...)` when used in EE expressions.
- For optional imports, add `# pyright: ignore[reportMissingImports]` on the import line.
- Do not add `.pyi` stubs — use inline annotations only.
- When a function accepts `GeoDataFrameLike`, require `crs` for raw `polars.DataFrame` inputs; use `normalize_geodataframe(..., crs=...)` to canonicalize.

Common fixes (short snippets):

- Casting `getInfo()`:

```py
from typing import cast

names: list[str] = cast(list[str], collection.aggregate_array("band").getInfo())
```

- Safe `fetchone()` usage:

```py
row = conn.execute(sa.text("SELECT id"), {...}).fetchone()
assert row is not None
value: int = row[0]
```

- Async gather handling:

```py
for cid, res in zip(batch, results):
    if isinstance(res, BaseException):
        handle_error(cid)
    else:
        handle_success(res)
```

Repo entrypoints:

```bash
# Typecheck
pixi run pyright
# Run tests
pixi run pytest
```

Add or update docs + tests when changing behaviour. Keep PRs small and reversible.

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
