import geopandas as gpd
import pandas as pd
import pytest

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import check_np_array_equivalence, get_all_satellites_for_test

all_satellites = get_all_satellites_for_test()


@pytest.mark.parametrize("satellite", all_satellites)
def test_download_images(satellite: AbstractSatellite) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite).to_numpy()
    original_sits = pd.read_parquet(f"tests/data/sits/0_{satellite.shortName}.parquet").to_numpy()
    assert check_np_array_equivalence(sits, original_sits)
