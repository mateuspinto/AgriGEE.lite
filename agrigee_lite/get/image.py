import logging
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Polygon
from tqdm.std import tqdm

from agrigee_lite.downloader import DownloaderStrategy
from agrigee_lite.ee_utils import ee_img_to_numpy
from agrigee_lite.misc import create_dict_hash, log_dict_function_call_summary
from agrigee_lite.sat.abstract_satellite import AbstractSatellite, SingleImageSatellite


def download_multiple_images(  # noqa: C901
    geometry: Polygon | MultiPolygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    invalid_images_threshold: float = 0.5,
    max_parallel_downloads: int = 40,
    force_redownload: bool = False,
):
    start_date = start_date.strftime("%Y-%m-%d") if isinstance(start_date, pd.Timestamp) else start_date
    end_date = end_date.strftime("%Y-%m-%d") if isinstance(end_date, pd.Timestamp) else end_date

    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    ee_feature = ee.Feature(
        ee_geometry,
        {"s": start_date, "e": end_date, "0": 1},
    )
    ee_expression = satellite.imageCollection(ee_feature)

    metadata_dict: dict[str, str] = {}
    metadata_dict |= log_dict_function_call_summary([
        "geometry",
        "start_date",
        "end_date",
        "satellite",
        "max_parallel_downloads",
        "force_redownload",
    ])
    metadata_dict |= satellite.log_dict()
    metadata_dict["start_date"] = start_date
    metadata_dict["end_date"] = end_date
    metadata_dict["centroid_x"] = geometry.centroid.x
    metadata_dict["centroid_y"] = geometry.centroid.y

    if ee_expression.size().getInfo() == 0:
        print("No images found for the specified parameters.")
        return np.array([]), []

    max_valid_pixels = ee_expression.aggregate_max("ZZ_USER_VALID_PIXELS")
    threshold = ee.Number(max_valid_pixels).multiply(invalid_images_threshold)
    ee_expression = ee_expression.filter(ee.Filter.gte("ZZ_USER_VALID_PIXELS", threshold))

    image_names = ee_expression.aggregate_array("ZZ_USER_TIME_DUMMY").getInfo()
    image_indexes = ee_expression.aggregate_array("system:index").getInfo()

    output_path = pathlib.Path("data/temp/images") / f"{create_dict_hash(metadata_dict)}"
    output_path.mkdir(parents=True, exist_ok=True)

    if force_redownload:
        for f in output_path.glob("*.zip"):
            f.unlink()

    downloader = DownloaderStrategy(download_folder=output_path)

    already_downloaded_files = {int(x.stem) for x in output_path.glob("*.zip")}
    all_chunks = set(range(len(image_indexes)))
    pending_chunks = sorted(all_chunks - already_downloaded_files)

    pbar = tqdm(total=len(pending_chunks), desc=f"Downloading images ({output_path.name})", unit="feature")

    def update_pbar():
        pbar.n = downloader.num_completed_downloads
        pbar.refresh()
        pbar.set_postfix({
            "aria2_errors": downloader.num_downloads_with_error,
            "active_downloads": downloader.num_unfinished_downloads,
        })

    def download_task(chunk_index):
        try:
            img = ee.Image(ee_expression.filter(ee.Filter.eq("system:index", image_indexes[chunk_index])).first())
            url = img.getDownloadURL({"name": str(chunk_index), "region": ee_geometry})
            downloader.add_download([(chunk_index, url)])
            return chunk_index, True  # noqa: TRY300
        except Exception as _:
            return chunk_index, False

    while downloader.num_completed_downloads < len(pending_chunks):
        with ThreadPoolExecutor(max_workers=max_parallel_downloads) as executor:
            futures = {executor.submit(download_task, chunk): chunk for chunk in pending_chunks}

            failed_chunks = []
            for future in as_completed(futures):
                chunk, success = future.result()
                if not success:
                    failed_chunks.append(chunk)
                    logging.warning(f"Download images - {output_path} - Failed to initiate download for chunk {chunk}.")

                update_pbar()

                while downloader.num_unfinished_downloads >= max_parallel_downloads:
                    time.sleep(1)
                    update_pbar()

        while downloader.num_unfinished_downloads > 0:
            time.sleep(1)
            update_pbar()

        pending_chunks = sorted(set(failed_chunks + downloader.failed_downloads))

    update_pbar()
    pbar.close()

    return image_names


def download_single_image(
    geometry: Polygon,
    satellite: SingleImageSatellite,
) -> np.ndarray:
    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    ee_feature = ee.Feature(ee_geometry, {"0": 1})

    try:
        image = satellite.image(ee_feature)
        image_clipped = image.clip(ee_geometry)
        image_np = ee_img_to_numpy(image_clipped, ee_geometry, satellite.pixelSize)
    except Exception as e:
        print(f"download_single_image_{satellite.shortName} = {e}")
        return np.array([])

    return image_np
