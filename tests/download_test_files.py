import os

import ee
import geopandas as gpd
import numpy as np

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import get_all_date_types_for_test, get_all_reducers_for_test, get_all_satellites_for_test


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

    all_reducers = get_all_reducers_for_test()

    for reducer in all_reducers:
        sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, reducers=[reducer])
        sits.to_parquet(f"tests/data/sits/0_{satellite.shortName}_{reducer}.parquet")


def download_for_test_download_multiple_sits() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(bands=["red"])
    sits = agl.get.multiple_sits(gdf.iloc[0:2], satellite, ["kurt", "median"], ["doy"], 0.7)
    sits.to_parquet("tests/data/sits/multithread.parquet")


# def download_for_test_download_multiple_sits_async() -> None:
#     from agrigee_lite.get.sits import __download_multiple_sits_async

#     gdf = gpd.read_parquet("tests/data/gdf.parquet")
#     satellite = agl.sat.Sentinel2(bands=["swir1", "nir"])
#     sits = anyio.run(
#         partial(__download_multiple_sits_async, gdf.iloc[0:2], satellite, ["skew", "p13"], ["doy"], 1),
#         backend_options={"use_uvloop": True},
#     )
#     sits.to_parquet("tests/data/sits/async.parquet")


def download_for_test_multiple_reducers() -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(bands=["swir1", "nir"])
    row = gdf.iloc[0]

    sits = agl.get.sits(
        row.geometry, row.start_date, row.end_date, satellite, ["mode", "p95", "p5", "var"], ["doy"], 0.3
    )
    sits.to_parquet("tests/data/sits/multiple_reducers.parquet")


def download_for_test_date_type(date_type: str) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(bands=["nir", "green"])
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, ["kurt", "mode"], [date_type], 100)
    sits.to_parquet(f"tests/data/sits/datetype_{date_type}.parquet")


def test_all_date_types(all_date_types: list[str]) -> None:
    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    satellite = agl.sat.Sentinel2(bands=["swir1", "swir2", "re4"])
    row = gdf.iloc[0]

    sits = agl.get.sits(row.geometry, row.start_date, row.end_date, satellite, ["kurt", "mode"], all_date_types, 200)
    sits.to_parquet("tests/data/sits/all_date_types.parquet")


if __name__ == "__main__":
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project="ee-paulagibrim")
    all_satellites = get_all_satellites_for_test()
    all_date_types = get_all_date_types_for_test()

    os.makedirs("tests/data/imgs", exist_ok=True)
    os.makedirs("tests/data/sits", exist_ok=True)

    for date_type in all_date_types:
        print("Downloading date type", date_type, "...")
        download_for_test_date_type(date_type)

    test_all_date_types(all_date_types)
    download_for_test_download_multiple_sits()
    # download_for_test_download_multiple_sits_async()
    download_for_test_multiple_reducers()

    for satellite in all_satellites:
        print("Downloading satellite", satellite.shortName, "...")
        download_img_for_test(satellite)
        download_sits_for_test(satellite)
        download_sits_for_test_with_reducers(satellite)
