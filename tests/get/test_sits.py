from functools import partial

import anyio
import geopandas as gpd
import pandas as pd
import pytest

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import assert_np_array_equivalence, get_all_satellites_for_test

all_satellites = get_all_satellites_for_test()
all_reducers = ["min", "max", "mean", "median", "std", "var", "p2", "p98", "kurt", "skew"]


@pytest.mark.parametrize("satellite", all_satellites)
def test_satellites_in_single_sits(satellite: AbstractSatellite) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite).to_numpy()
    original_sits = pd.read_parquet(f"tests/data/sits/0_{satellite.shortName}.parquet").to_numpy()
    assert_np_array_equivalence(sits, original_sits)


@pytest.mark.parametrize("satellite", all_satellites)
@pytest.mark.parametrize("reducer", all_reducers)
def test_reducers_of_all_satellites_in_single_sits(satellite: AbstractSatellite, reducer: str) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, reducers=[reducer]).to_numpy()
    original_sits = pd.read_parquet(f"tests/data/sits/0_{satellite.shortName}_{reducer}.parquet").to_numpy()
    assert_np_array_equivalence(sits.squeeze(), original_sits.squeeze())


def test_download_multiple_sits() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(selected_bands=["red"])
    sits = agl.get.multiple_sits(gdf.iloc[0:2], satellite, ["kurt", "median"], 0.7).to_numpy()
    original_sits = pd.read_parquet("tests/data/sits/multiplesits.parquet").to_numpy()

    assert_np_array_equivalence(sits.squeeze(), original_sits.squeeze())


def test_download_multiple_sits_multithread() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(selected_bands=["swir1", "nir"])
    sits = agl.get.multiple_sits_multithread(gdf.iloc[0:2], satellite, ["skew", "p13"], 0.3).to_numpy()
    original_sits = pd.read_parquet("tests/data/sits/multithread.parquet").to_numpy()

    assert_np_array_equivalence(sits.squeeze(), original_sits.squeeze())


def test_download_multiple_sits_async() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(selected_bands=["swir1", "nir"])
    sits = anyio.run(
        partial(agl.get.multiple_sits_async, gdf.iloc[0:2], satellite, ["skew", "p13"], 0.3),
        backend_options={"use_uvloop": True},
    ).to_numpy()
    original_sits = pd.read_parquet("tests/data/sits/async.parquet").to_numpy()

    assert_np_array_equivalence(sits.squeeze(), original_sits.squeeze())
