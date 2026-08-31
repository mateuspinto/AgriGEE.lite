"""
Behaviour tests for MapBiomasPastureVigor.

Like the rest of tests/get, these hit Earth Engine.  The product only has
values where MapBiomas maps pasture, so the fixtures use a known pasture
polygon in the Cerrado and the bundled (soybean) sample as the negative case.
"""

import polars as pl
import pytest
from shapely.geometry import box

import agrigee_lite as agl

# Pasture in Goiás, mapped as pasture in every year of the product.
PASTURE_GEOMETRY = box(-49.60, -16.90, -49.55, -16.85)

VIGOR_COLUMNS = ["lowVigor", "mediumVigor", "highVigor"]

FIRST_MAPPED_YEAR = 2000
LAST_MAPPED_YEAR = 2024


def _years(df: pl.DataFrame) -> list[int]:
    return df.get_column("timestamp").dt.year().to_list()


@pytest.fixture(scope="module")
def vigor_series() -> pl.DataFrame:
    return agl.get.sits(PASTURE_GEOMETRY, "2018-01-01", "2024-12-31", agl.sat.MapBiomasPastureVigor())


def test_returns_one_row_per_requested_year(vigor_series: pl.DataFrame) -> None:
    assert _years(vigor_series) == list(range(2018, 2025))


def test_returns_a_fraction_per_vigor_class(vigor_series: pl.DataFrame) -> None:
    assert set(VIGOR_COLUMNS).issubset(vigor_series.columns)


def test_fractions_sum_to_one(vigor_series: pl.DataFrame) -> None:
    totals = vigor_series.select(pl.sum_horizontal(VIGOR_COLUMNS)).to_series().to_list()
    assert all(abs(total - 1) < 1e-6 for total in totals), totals


def test_fractions_stay_within_zero_and_one(vigor_series: pl.DataFrame) -> None:
    fractions = vigor_series.select(VIGOR_COLUMNS).to_numpy()
    assert ((fractions >= 0) & (fractions <= 1)).all()


def test_pasture_pixel_count_is_positive(vigor_series: pl.DataFrame) -> None:
    assert all(count > 0 for count in vigor_series.get_column("validPixelsCount").to_list())


def test_years_outside_coverage_are_clipped_away() -> None:
    series = agl.get.sits(PASTURE_GEOMETRY, "1995-01-01", "2030-12-31", agl.sat.MapBiomasPastureVigor())
    assert _years(series) == list(range(FIRST_MAPPED_YEAR, LAST_MAPPED_YEAR + 1))


def test_geometry_that_is_not_pasture_returns_no_rows() -> None:
    soybean_field = agl.get_sample_gdf().iloc[0].geometry
    series = agl.get.sits(soybean_field, "2018-01-01", "2024-12-31", agl.sat.MapBiomasPastureVigor())
    assert series.is_empty()
