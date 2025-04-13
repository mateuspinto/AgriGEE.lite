import ee
import geopandas as gpd
import numpy as np

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import get_all_satellites_for_test


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


if __name__ == "__main__":
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project="ee-paulagibrim")

    all_satellites = get_all_satellites_for_test()
    for satellite in all_satellites:
        print("Downloading satellite", satellite.shortName, "...")
        download_img_for_test(satellite)
        download_sits_for_test(satellite)
