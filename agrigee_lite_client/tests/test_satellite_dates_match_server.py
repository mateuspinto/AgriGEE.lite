"""Cross-checks the embedded satellite date table against the server's own
REGISTRY — catches drift whenever `agrigee_lite` happens to be importable
alongside this package (i.e. in this monorepo's dev environment). A real
end-user installing just agrigee_lite_client won't have `agrigee_lite`
installed, so this is skipped for them; it's meant for our own CI/dev use.

If this test fails: regenerate the table
(scripts/generate_satellite_dates.py) and bump both packages' version
together (SPECS.md §2.1).
"""

from __future__ import annotations

import pytest

from agrigee_lite_client._satellite_dates import SATELLITE_DATE_RANGES, get_satellite_date_range

agrigee_lite = pytest.importorskip("agrigee_lite", reason="only runs inside the AgriGEE.lite monorepo dev env")

from agrigee_lite.api._satellites import REGISTRY  # noqa: E402


def _server_date_range(name: str) -> tuple[str, str] | None:
    try:
        instance = REGISTRY[name]()
    except TypeError:
        return None  # e.g. TwoSatelliteFusion — needs more than defaults, excluded on purpose
    return (instance.startDate, instance.endDate)


def test_every_table_entry_matches_the_live_satellite_class() -> None:
    mismatches = []
    for name, expected in SATELLITE_DATE_RANGES.items():
        actual = _server_date_range(name)
        if actual != expected:
            mismatches.append((name, expected, actual))
    assert not mismatches, f"stale entries (name, table, live): {mismatches}"


def test_no_registered_satellite_is_missing_from_the_table() -> None:
    instantiable = {name for name in REGISTRY if _server_date_range(name) is not None}
    missing = instantiable - set(SATELLITE_DATE_RANGES)
    assert not missing, f"satellites missing from the client table: {sorted(missing)}"


def test_sentinel2_use_sr_false_matches_the_live_class() -> None:
    from agrigee_lite.sat.sentinel2 import Sentinel2

    live = Sentinel2(use_sr=False)
    assert get_satellite_date_range("Sentinel2", {"use_sr": False}) == (live.startDate, live.endDate)
