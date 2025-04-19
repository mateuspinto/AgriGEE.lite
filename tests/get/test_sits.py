import geopandas as gpd
import pandas as pd
import pytest

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import assert_np_array_equivalence, get_all_satellites_for_test

all_satellites = get_all_satellites_for_test()
all_reducers = ["min", "max", "mean", "median", "std", "var", "p2", "p98", "kurt", "skew"]


@pytest.mark.parametrize("satellite", all_satellites)
def test_download_sits(satellite: AbstractSatellite) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite).to_numpy()
    original_sits = pd.read_parquet(f"tests/data/sits/0_{satellite.shortName}.parquet").to_numpy()
    assert_np_array_equivalence(sits, original_sits)


@pytest.mark.parametrize("satellite", all_satellites)
@pytest.mark.parametrize("reducer", all_reducers)
def test_download_sits_with_reducer(satellite: AbstractSatellite, reducer: str) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, reducers=[reducer]).to_numpy()
    original_sits = pd.read_parquet(f"tests/data/sits/0_{satellite.shortName}_{reducer}.parquet").to_numpy()
    assert_np_array_equivalence(sits.squeeze(), original_sits.squeeze())
