"""
Behaviour tests for the MapBiomas Collection 11 data sources.

Like the rest of tests/get, these hit Earth Engine.  The fixtures use a
double-cropped soybean field in Mato Grosso, which is inside the second crop
module's ten-state extent and carries a safrinha in every recent year, so the
assertions can be about shape and magnitude rather than exact values, which
shift between product versions.
"""

import polars as pl
import pytest
from shapely.geometry import box

import agrigee_lite as agl

# Soy/corn field near Sinop, MT: double-cropped, and inside every module's extent.
FIELD_GEOMETRY = box(-55.02, -13.02, -54.98, -12.98)

# Carajás, PA — the largest iron ore mine in the world, for the mining module.
MINE_GEOMETRY = box(-50.10, -6.15, -49.90, -5.95)

SECOND_CROP_COLUMNS = [
    "noSecondCrop",
    "cornSecondCrop",
    "cottonSecondCrop",
    "otherTemporarySecondCrop",
]
CYCLE_COLUMNS = ["oneCycle", "twoCycles", "threeOrMoreCycles"]


def _years(df: pl.DataFrame) -> list[int]:
    return df.get_column("timestamp").dt.year().to_list()


@pytest.fixture(scope="module")
def second_crop() -> pl.DataFrame:
    return agl.get.sits(FIELD_GEOMETRY, "2018-01-01", "2024-12-31", agl.sat.MapBiomasSecondCrop())


@pytest.fixture(scope="module")
def crop_cycles() -> pl.DataFrame:
    return agl.get.sits(FIELD_GEOMETRY, "2018-01-01", "2024-12-31", agl.sat.MapBiomasCropCycles())


def test_second_crop_returns_one_row_per_requested_year(second_crop: pl.DataFrame) -> None:
    assert _years(second_crop) == list(range(2018, 2025))


def test_second_crop_fractions_sum_to_one(second_crop: pl.DataFrame) -> None:
    totals = second_crop.select(pl.sum_horizontal(SECOND_CROP_COLUMNS)).to_series().to_list()
    assert all(abs(total - 1) < 1e-6 for total in totals), totals


def test_second_crop_on_a_double_cropped_field_is_mostly_corn(second_crop: pl.DataFrame) -> None:
    """Corn is overwhelmingly the safrinha crop in Mato Grosso."""
    assert second_crop.get_column("cornSecondCrop").min() > 0.5


def test_second_crop_before_the_product_starts_is_empty() -> None:
    """The product begins in 2000; earlier years are clipped away, not clamped."""
    df = agl.get.sits(FIELD_GEOMETRY, "1990-01-01", "1995-12-31", agl.sat.MapBiomasSecondCrop())
    assert df.is_empty()


def test_crop_cycles_returns_one_row_per_requested_year(crop_cycles: pl.DataFrame) -> None:
    assert _years(crop_cycles) == list(range(2018, 2025))


def test_crop_cycles_fractions_sum_to_one(crop_cycles: pl.DataFrame) -> None:
    totals = crop_cycles.select(pl.sum_horizontal(CYCLE_COLUMNS)).to_series().to_list()
    assert all(abs(total - 1) < 1e-6 for total in totals), totals


def test_crop_cycles_before_2017_is_empty() -> None:
    """Unlike the other MapBiomas sources this one is Sentinel-derived and starts in 2017."""
    df = agl.get.sits(FIELD_GEOMETRY, "2010-01-01", "2015-12-31", agl.sat.MapBiomasCropCycles())
    assert df.is_empty()


def test_mean_crop_cycles_agrees_with_the_annual_product(crop_cycles: pl.DataFrame) -> None:
    """The static mean layer should land near the annual fractions' own mean."""
    mean_df = agl.get.sits(FIELD_GEOMETRY, "2018-01-01", "2024-12-31", agl.sat.MapBiomasCropCyclesMean())
    published_mean = mean_df.get_column("meanCropCycles").to_list()[0]

    annual_mean = (
        crop_cycles.select(
            pl.col("oneCycle") + pl.col("twoCycles") * 2 + pl.col("threeOrMoreCycles") * 3
        )
        .to_series()
        .mean()
    )
    assert abs(published_mean - annual_mean) < 0.5, (published_mean, annual_mean)


def test_pasture_age_reports_ages_within_the_series_length() -> None:
    """Age is encoded as 200 + years since 1985, so it cannot exceed the series length."""
    df = agl.get.sits(FIELD_GEOMETRY, "2020-01-01", "2024-12-31", agl.sat.MapBiomasPastureAge())
    if df.is_empty():
        pytest.skip("no pasture mapped in this geometry")

    ages = df.get_column("meanPastureAge").to_list()
    assert all(0 <= age <= 40 for age in ages), ages


def test_pasture_age_fractions_sum_to_one() -> None:
    df = agl.get.sits(FIELD_GEOMETRY, "2020-01-01", "2024-12-31", agl.sat.MapBiomasPastureAge())
    if df.is_empty():
        pytest.skip("no pasture mapped in this geometry")

    totals = (
        df.select(pl.sum_horizontal(["agedPixelsFraction", "undatedPixelsFraction"])).to_series().to_list()
    )
    assert all(abs(total - 1) < 1e-6 for total in totals), totals


def test_coverage_on_a_soybean_field_is_classified_as_soybean() -> None:
    """Class 39 is Soybean in the MapBiomas legend."""
    df = agl.get.sits(FIELD_GEOMETRY, "2020-01-01", "2024-12-31", agl.sat.MapBiomasC11())
    assert 39 in df.get_column("class").to_list()


def test_mining_reports_iron_at_carajas() -> None:
    """Carajás is an iron ore mine, and 102 is iron in the substance hierarchy."""
    df = agl.get.sits(MINE_GEOMETRY, "2020-01-01", "2022-12-31", agl.sat.MapBiomasMining())
    assert not df.is_empty()
    assert all(str(code).endswith("102") for code in df.get_column("class").to_list())


def test_mining_is_empty_where_there_is_no_mining() -> None:
    df = agl.get.sits(FIELD_GEOMETRY, "2020-01-01", "2022-12-31", agl.sat.MapBiomasMining())
    assert df.is_empty()
