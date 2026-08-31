"""
Behaviour tests for ChirpsAnnualRainfall.

Like the rest of tests/get, these hit Earth Engine.  The fixtures use a
Cerrado polygon with a well-known rainfall regime (~1200-1800 mm/yr), so the
assertions can be about magnitude and shape rather than exact values, which
vary with the CHIRPS release.
"""

import polars as pl
import pytest
from shapely.geometry import box

import agrigee_lite as agl

# Pasture region in Goiás — same area the pasture-vigor tests use, so the two
# products can be compared over one geometry.
CERRADO_GEOMETRY = box(-49.60, -16.90, -49.55, -16.85)

FIRST_MAPPED_YEAR = 1981


def _years(df: pl.DataFrame) -> list[int]:
    return df.get_column("timestamp").dt.year().to_list()


@pytest.fixture(scope="module")
def rainfall_series() -> pl.DataFrame:
    return agl.get.sits(CERRADO_GEOMETRY, "2015-01-01", "2024-12-31", agl.sat.ChirpsAnnualRainfall())


def test_returns_one_row_per_requested_year(rainfall_series: pl.DataFrame) -> None:
    assert _years(rainfall_series) == list(range(2015, 2025))


def test_returns_an_annual_total_not_a_daily_value(rainfall_series: pl.DataFrame) -> None:
    # Ten years of daily CHIRPS would be ~3650 rows; the point of this source is
    # that a decade costs ten.
    totals = rainfall_series.get_column("rainfallMm").to_list()
    assert len(totals) == 10
    assert all(600 < total < 3000 for total in totals), totals


def test_a_single_year_request_returns_that_year() -> None:
    series = agl.get.sits(CERRADO_GEOMETRY, "2020-01-01", "2020-12-31", agl.sat.ChirpsAnnualRainfall())
    assert _years(series) == [2020]


def test_years_before_coverage_are_clipped_away() -> None:
    series = agl.get.sits(CERRADO_GEOMETRY, "1975-01-01", "1983-12-31", agl.sat.ChirpsAnnualRainfall())
    assert min(_years(series)) >= FIRST_MAPPED_YEAR


def test_a_geometry_smaller_than_one_chirps_cell_still_returns_a_value() -> None:
    # CHIRPS cells are ~5.5 km; most paddocks are smaller, and bestEffort must
    # keep them from coming back empty.
    tiny = box(-49.58, -16.88, -49.578, -16.878)
    series = agl.get.sits(tiny, "2022-01-01", "2022-12-31", agl.sat.ChirpsAnnualRainfall())
    assert series.get_column("rainfallMm").to_list()[0] > 0


def test_valid_pixel_count_reports_the_reduction_size(rainfall_series: pl.DataFrame) -> None:
    assert all(count > 0 for count in rainfall_series.get_column("validPixelsCount").to_list())


def test_rainfall_varies_between_years(rainfall_series: pl.DataFrame) -> None:
    # A constant column would mean the yearly filter is not actually slicing the
    # collection — the failure mode a per-year loop invites.
    totals = rainfall_series.get_column("rainfallMm").to_list()
    assert len({round(total) for total in totals}) > 1, totals
