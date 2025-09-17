import logging
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Point, Polygon
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
    image_indices: list[int] | None = None,
) -> list[str]:
    """
    Download multiple satellite images for a given geometry and date range.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon
        The area of interest as a shapely Polygon or MultiPolygon.
    start_date : pd.Timestamp or str
        Start date for image collection.
    end_date : pd.Timestamp or str
        End date for image collection.
    satellite : AbstractSatellite
        The satellite configuration to use for image collection.
    invalid_images_threshold : float, optional
        Threshold for filtering images based on valid pixels (0.0-1.0), by default 0.5.
    max_parallel_downloads : int, optional
        Maximum number of parallel downloads, by default 40.
    force_redownload : bool, optional
        Whether to force re-download of existing files, by default False.
    image_indices : list[int] or None, optional
        List of specific image indices to download (e.g., [0, 1] for first two images).
        If None, all images in the date range will be downloaded, by default None.

    Returns
    -------
    list[str]
        List of image names (dates in YYYY-MM-DD format) that were downloaded.
    """

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

    # Filter images by indices if provided
    if image_indices is not None:
        # Ensure indices are valid
        valid_indices = [i for i in image_indices if 0 <= i < len(image_indexes)]
        if not valid_indices:
            print("No valid image indices provided.")
            return np.array([]), []

        image_names = [image_names[i] for i in valid_indices]
        image_indexes = [image_indexes[i] for i in valid_indices]

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
            # Use only the image date as filename (GEE standard format)
            image_date = image_names[chunk_index]
            filename = f"{image_date}"
            url = img.getDownloadURL({"name": filename, "region": ee_geometry})
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
    geometry: Polygon | MultiPolygon | Point,
    satellite: SingleImageSatellite,
) -> np.ndarray:
    """
    Download a single satellite image for a given geometry.

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, or Point
        The area or point of interest for image extraction.
    satellite : SingleImageSatellite
        The satellite configuration object for single image extraction.

    Returns
    -------
    np.ndarray
        NumPy array containing the satellite image data. Returns empty array if download fails.
    """
    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    ee_feature = ee.Feature(ee_geometry, {"0": 1})

    try:
        image = satellite.image(ee_feature)
        image_clipped = image.clip(ee_geometry)
        image_np = ee_img_to_numpy(image_clipped, ee_geometry, satellite.pixelSize)
    except Exception:
        logging.exception(f"Failed to download single image for satellite {satellite.shortName}")
        return np.array([])

    return image_np
