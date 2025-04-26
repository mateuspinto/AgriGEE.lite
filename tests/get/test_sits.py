from functools import partial

import anyio
import geopandas as gpd
import pandas as pd
import pytest

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import assert_df_equivalence, get_all_satellites_for_test

all_satellites = get_all_satellites_for_test()
all_reducers = ["min", "max", "mean", "median", "std", "var", "p2", "p98", "kurt", "skew"]
all_date_types = ["doy", "year", "fyear"]


@pytest.mark.parametrize("satellite", all_satellites)
def test_satellites_in_single_sits(satellite: AbstractSatellite) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite)
    original_sits = pd.read_parquet(f"tests/data/sits/0_{satellite.shortName}.parquet")
    assert_df_equivalence(sits, original_sits)


@pytest.mark.parametrize("satellite", all_satellites)
@pytest.mark.parametrize("reducer", all_reducers)
def test_reducers_of_all_satellites_in_single_sits(satellite: AbstractSatellite, reducer: str) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, reducers=[reducer])
    original_sits = pd.read_parquet(f"tests/data/sits/0_{satellite.shortName}_{reducer}.parquet")
    assert_df_equivalence(sits, original_sits)


def test_multiple_reducers() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(bands=["swir1", "nir"])
    row = gdf.iloc[0]

    sits = agl.get.sits(
        row.geometry, row.start_date, row.end_date, satellite, ["mode", "p95", "p5", "var"], ["doy"], 0.3
    )
    original_sits = pd.read_parquet("tests/data/sits/multiple_reducers.parquet")

    assert_df_equivalence(sits, original_sits)


def test_download_multiple_sits_multithread() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(bands=["swir1", "nir"])
    sits = agl.get.multiple_sits(gdf.iloc[0:2], satellite, ["skew", "p13"], ["doy"], 0.3)
    original_sits = pd.read_parquet("tests/data/sits/multithread.parquet")

    assert_df_equivalence(sits, original_sits)


@pytest.mark.parametrize("date_type", all_date_types)
def test_date_type(date_type: str) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(bands=["nir", "green"])
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, ["kurt", "mode"], [date_type], 100)
    original_sits = pd.read_parquet(f"tests/data/sits/datetype_{date_type}.parquet")

    assert_df_equivalence(sits, original_sits)


def test_all_date_types() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(bands=["swir1", "swir2", "re4"])
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, ["kurt", "mode"], all_date_types, 200)
    original_sits = pd.read_parquet("tests/data/sits/all_date_types.parquet")

    assert_df_equivalence(sits, original_sits)


# def test_download_multiple_sits_async() -> None:
#     from agrigee_lite.get.sits import __download_multiple_sits_async

#     gdf = gpd.read_parquet("tests/data/gdf.parquet")
#     satellite = agl.sat.Sentinel2(bands=["swir1", "nir"])
#     sits = anyio.run(
#         partial(__download_multiple_sits_async, gdf.iloc[0:2].copy(), satellite, ["skew", "p13"], ["doy"], 1),
#         backend_options={"use_uvloop": True},
#     )
#     original_sits = pd.read_parquet("tests/data/sits/async.parquet")

#     assert_df_equivalence(sits, original_sits)
