import geopandas as gpd
import numpy as np
import pytest

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import assert_np_array_equivalence, get_all_satellites_for_test

all_satellites = get_all_satellites_for_test()


@pytest.mark.parametrize("satellite", all_satellites)
def test_download_images(satellite: AbstractSatellite) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    imgs = agl.get.images(row.geometry, row.start_date, row.end_date, satellite)
    original_imgs = np.load(f"tests/data/imgs/0_{satellite.shortName}.npz")
    assert_np_array_equivalence(imgs, original_imgs["data"], 0)
