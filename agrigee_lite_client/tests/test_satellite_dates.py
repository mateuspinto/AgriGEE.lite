from __future__ import annotations

import pytest

from agrigee_lite_client._exceptions import AgriGEEUnknownSatelliteError
from agrigee_lite_client._satellite_dates import SATELLITE_DATE_RANGES, get_satellite_date_range


def test_known_satellite_returns_its_range() -> None:
    assert get_satellite_date_range("Sentinel1GRD") == ("2014-10-03", "2050-01-01")


def test_sentinel2_default_is_surface_reflectance_range() -> None:
    assert get_satellite_date_range("Sentinel2") == ("2019-01-01", "2050-01-01")
    assert get_satellite_date_range("Sentinel2", {}) == ("2019-01-01", "2050-01-01")
    assert get_satellite_date_range("Sentinel2", {"use_sr": True}) == ("2019-01-01", "2050-01-01")


def test_sentinel2_use_sr_false_extends_range() -> None:
    assert get_satellite_date_range("Sentinel2", {"use_sr": False}) == ("2016-01-01", "2050-01-01")


def test_unknown_satellite_raises() -> None:
    with pytest.raises(AgriGEEUnknownSatelliteError, match="NotASatellite"):
        get_satellite_date_range("NotASatellite")


def test_two_satellite_fusion_is_not_in_the_table() -> None:
    """Its constructor takes satellite objects, not JSON params — never reachable via {name, params}."""
    assert "TwoSatelliteFusion" not in SATELLITE_DATE_RANGES
