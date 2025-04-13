import ee
import geopandas as gpd
import numpy as np

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from tests.utils import get_all_satellites_for_test


def download_img_for_test(satellite: AbstractSatellite) -> None:
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project="ee-paulagibrim")

    gdf = gpd.read_parquet("tests/data/gdf.parquet")
    row = gdf.iloc[0]

    imgs = agl.get.images(row.geometry, row.start_date, row.end_date, satellite)
    np.savez_compressed(f"tests/data/imgs/0_{satellite.shortName}.npz", data=imgs)


if __name__ == "__main__":
    all_satellites = get_all_satellites_for_test()
    for satellite in all_satellites:
        print("Downloading satellite", satellite.shortName, "...")
        download_img_for_test(satellite)
