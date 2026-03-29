"""
Sensor-coverage tests for agl.get.sits().

One test per sensor — validates that the result is a non-empty DataFrame
with a NaN fraction below the threshold defined for each sensor type.
"""

import pytest

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import assert_sits_quality

# (satellite, start_date_override, end_date_override)
# None → use dates from the bundled sample GDF
_PARAMS: list[tuple[AbstractSatellite, str | None, str | None]] = [
    (agl.sat.Sentinel2(), None, None),
    (agl.sat.Sentinel2(use_sr=False), None, None),
    (agl.sat.Sentinel1GRD(), None, None),
    (agl.sat.Sentinel1GRD(ascending=False), None, None),
    (agl.sat.Landsat9(), None, None),
    (agl.sat.Landsat8(), None, None),
    (agl.sat.Landsat7(), "2020-01-01", "2020-12-31"),
    (agl.sat.Landsat5(), "2010-01-01", "2010-12-31"),
    (agl.sat.HLSSentinel2(), None, None),
    (agl.sat.HLSLandsat(), None, None),
    (agl.sat.ModisDaily(), None, None),
    (agl.sat.Modis8Days(), None, None),
    (agl.sat.PALSAR2ScanSAR(), None, None),
    (agl.sat.MapBiomas(), None, None),
    (agl.sat.SatelliteEmbedding(), None, None),
    (agl.sat.ANADEM(), None, None),
    (agl.sat.CopernicusDEM(), None, None),
    (agl.sat.WRBSoilClasses(), None, None),
]


@pytest.fixture(scope="module")
def sample_row():
    return agl.get_sample_gdf().iloc[0]


@pytest.mark.parametrize(
    "satellite,start_override,end_override",
    _PARAMS,
    ids=[p[0].shortName for p in _PARAMS],
)
def test_sits_data_quality(
    satellite: AbstractSatellite,
    start_override: str | None,
    end_override: str | None,
    sample_row,
) -> None:
    start = start_override or sample_row.start_date
    end = end_override or sample_row.end_date

    result = agl.get.sits(sample_row.geometry, start, end, satellite)

    assert_sits_quality(result, satellite)
