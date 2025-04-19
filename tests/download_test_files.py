from functools import partial

import anyio
import ee
import geopandas as gpd
import numpy as np

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import get_all_satellites_for_test

all_reducers = ["min", "max", "mean", "median", "std", "var", "p2", "p98", "kurt", "skew"]


def download_img_for_test(satellite: AbstractSatellite) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    imgs = agl.get.images(row.geometry, row.start_date, row.end_date, satellite)
    np.savez_compressed(f"tests/data/imgs/0_{satellite.shortName}.npz", data=imgs)


def download_sits_for_test(satellite: AbstractSatellite) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite)
    sits.to_parquet(f"tests/data/sits/0_{satellite.shortName}.parquet")


def download_sits_for_test_with_reducers(satellite: AbstractSatellite) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    for reducer in all_reducers:
        sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, reducers=[reducer])
        sits.to_parquet(f"tests/data/sits/0_{satellite.shortName}_{reducer}.parquet")


def download_for_test_download_multiple_sits() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(selected_bands=["red"])
    sits = agl.get.multiple_sits(gdf.iloc[0:2], satellite, ["kurt", "median"], 0.7)
    sits.to_parquet("tests/data/sits/multiplesits.parquet")


def download_for_test_download_multiple_sits_multithread() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(selected_bands=["swir1", "nir"])
    sits = agl.get.multiple_sits_multithread(gdf.iloc[0:2], satellite, ["skew", "p13"], 0.3)
    sits.to_parquet("tests/data/sits/multithread.parquet")


def download_for_test_download_multiple_sits_async() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(selected_bands=["swir1", "nir"])
    sits = anyio.run(
        partial(agl.get.multiple_sits_async, gdf.iloc[0:2], satellite, ["skew", "p13"], 0.3),
        backend_options={"use_uvloop": True},
    )
    sits.to_parquet("tests/data/sits/async.parquet")


if __name__ == "__main__":
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project="ee-paulagibrim")

    download_for_test_download_multiple_sits()
    download_for_test_download_multiple_sits_multithread()
    download_for_test_download_multiple_sits_async()

    all_satellites = get_all_satellites_for_test()
    for satellite in all_satellites:
        print("Downloading satellite", satellite.shortName, "...")
        download_img_for_test(satellite)
        download_sits_for_test(satellite)
        download_sits_for_test_with_reducers(satellite)
