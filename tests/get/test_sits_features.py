"""
Feature tests for agl.get.sits() and agl.get.multiple_sits().

Covers: default reducer, all individual reducers, combined reducers,
all date encodings, and multi-geometry parallel download.
"""

import pytest
import polars as pl
import polars.selectors as cs

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import assert_sits_quality, get_all_reducers_for_test, get_all_satellites_for_test

_SATELLITES = get_all_satellites_for_test()
_REDUCERS = get_all_reducers_for_test()
_DATE_TYPES = ["doy", "year", "fyear"]


@pytest.fixture(scope="module")
def sample_gdf():
    return agl.get_sample_gdf()


# ---------------------------------------------------------------------------
# Default reducer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("satellite", _SATELLITES, ids=lambda s: s.shortName)
def test_default_sits(satellite: AbstractSatellite, sample_gdf) -> None:
    row = sample_gdf.iloc[0]
    result = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite)
    assert isinstance(result, pl.DataFrame)
    assert_sits_quality(result, satellite)


# ---------------------------------------------------------------------------
# Individual reducers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("satellite", _SATELLITES, ids=lambda s: s.shortName)
@pytest.mark.parametrize("reducer", _REDUCERS)
def test_single_reducer(satellite: AbstractSatellite, reducer: str, sample_gdf) -> None:
    row = sample_gdf.iloc[0]
    result = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, reducers=[reducer])
    assert isinstance(result, pl.DataFrame)
    assert_sits_quality(result, satellite)


# ---------------------------------------------------------------------------
# Combined reducers
# ---------------------------------------------------------------------------


def test_combined_reducers(sample_gdf) -> None:
    satellite = agl.sat.Sentinel2(bands=["swir1", "nir"])
    row = sample_gdf.iloc[0]
    reducers = ["mode", "p95", "p5", "var"]

    result = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, reducers, ["doy"], 0.3)

    assert_sits_quality(result, satellite)
    # More reducers → more columns than a single-reducer call
    single = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, ["median"])
    assert result.shape[1] > single.shape[1], "Combined reducers should produce more columns than a single reducer"


# ---------------------------------------------------------------------------
# Date encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("date_type", _DATE_TYPES)
def test_date_encoding(date_type: str, sample_gdf) -> None:
    satellite = agl.sat.Sentinel2(bands=["nir", "green"])
    row = sample_gdf.iloc[0]

    result = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, ["median"], [date_type], 100)

    assert_sits_quality(result, satellite)
    numeric = result.select(cs.numeric())
    assert numeric.width > 0, f"Expected numeric columns for date_type={date_type!r}"


def test_all_date_types_combined(sample_gdf) -> None:
    satellite = agl.sat.Sentinel2(bands=["swir1", "swir2", "re4"])
    row = sample_gdf.iloc[0]

    single_type = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, ["median"], ["doy"])
    all_types = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, ["median"], _DATE_TYPES, 200)

    assert_sits_quality(all_types, satellite)
    assert (
        all_types.shape[1] > single_type.shape[1]
    ), "Using all date types should produce more columns than a single date type"


# ---------------------------------------------------------------------------
# Multiple geometries
# ---------------------------------------------------------------------------


def test_multiple_geometries(sample_gdf) -> None:
    satellite = agl.sat.Sentinel2(bands=["swir1", "nir"])
    gdf = sample_gdf.iloc[0:2]

    result = agl.get.multiple_sits(gdf, satellite, ["skew", "p13"], ["doy"], 0.3)

    assert isinstance(result, pl.DataFrame)
    assert_sits_quality(result, satellite)
