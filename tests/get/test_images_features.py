"""
Feature tests for agl.get.images().

Downloads only the first available image per sensor to keep tests fast,
then validates the returned date list.
"""

import re

import pytest

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import get_all_satellites_for_test

_SATELLITES = get_all_satellites_for_test()
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@pytest.fixture(scope="module")
def sample_row():
    return agl.get_sample_gdf().iloc[0]


@pytest.mark.parametrize("satellite", _SATELLITES, ids=lambda s: s.shortName)
def test_download_images(satellite: AbstractSatellite, sample_row) -> None:
    result = agl.get.images(
        sample_row.geometry,
        sample_row.start_date,
        sample_row.end_date,
        satellite,
        image_indices=[0],
    )

    assert isinstance(result, list), f"[{satellite.shortName}] Expected list, got {type(result)}"
    if len(result) == 0:
        pytest.skip(f"[{satellite.shortName}] No images found for the given geometry and date range")
    assert all(_DATE_RE.match(d) for d in result), f"[{satellite.shortName}] Unexpected date format in result: {result}"
