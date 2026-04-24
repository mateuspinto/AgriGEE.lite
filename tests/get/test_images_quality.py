"""
Sensor-coverage tests for agl.get.images() and agl.get.image().

imageCollection sensors: downloads only the first available image (image_indices=[0])
and validates the returned date string.

SingleImageSatellite sensors: downloads the static raster and validates that the
numpy array is non-empty with an acceptable NaN fraction.
"""

import re

import pytest

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite, SingleImageSatellite
from tests.utils import assert_single_image_quality

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# imageCollection-based sensors: (satellite, start_override, end_override)
# None → use dates from the bundled sample GDF
_IMAGES_PARAMS: list[tuple[AbstractSatellite, str | None, str | None]] = [
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
    (agl.sat.SatelliteEmbedding(), None, None),
]

# Static rasters — no date range needed
_SINGLE_IMAGE_SATELLITES: list[SingleImageSatellite] = [
    agl.sat.ANADEM(),
    agl.sat.CopernicusDEM(),
    agl.sat.WRBSoilClasses(),
]


@pytest.fixture(scope="module")
def sample_row():
    return agl.get_sample_gdf().iloc[0]


@pytest.mark.parametrize(
    "satellite,start_override,end_override",
    _IMAGES_PARAMS,
    ids=[p[0].shortName for p in _IMAGES_PARAMS],
)
def test_images_returns_valid_date(
    satellite: AbstractSatellite,
    start_override: str | None,
    end_override: str | None,
    sample_row,
) -> None:
    start = start_override or sample_row.start_date
    end = end_override or sample_row.end_date

    result = agl.get.images(sample_row.geometry, start, end, satellite, image_indices=[0])

    assert isinstance(result, list), f"[{satellite.shortName}] Expected list, got {type(result)}"
    if len(result) == 0:
        pytest.skip(f"[{satellite.shortName}] No images found for the given geometry and date range")
    assert all(_DATE_RE.match(d) for d in result), f"[{satellite.shortName}] Unexpected value in date list: {result}"


@pytest.mark.parametrize("satellite", _SINGLE_IMAGE_SATELLITES, ids=lambda s: s.shortName)
def test_single_image_data_quality(satellite: SingleImageSatellite, sample_row) -> None:
    result = agl.get.image(sample_row.geometry, satellite)
    assert_single_image_quality(result, satellite)
