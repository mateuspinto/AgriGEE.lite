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
