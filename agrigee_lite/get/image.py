import concurrent.futures
import logging
import logging.handlers
import queue
from functools import partial

import ee
import numpy as np
import pandas as pd
from shapely import Polygon
from tqdm.std import tqdm

from agrigee_lite.ee_utils import ee_img_to_numpy
from agrigee_lite.misc import cached
from agrigee_lite.sat.abstract_satellite import AbstractSatellite, SingleImageSatellite


# @cached
def download_multiple_images(
    geometry: Polygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    invalid_images_threshold: float = 0.5,
    num_threads_rush: int = 30,
    num_threads_retry: int = 10,
) -> np.ndarray:
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)

    file_handler = logging.FileHandler("logging.log", mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    queue_listener = logging.handlers.QueueListener(log_queue, file_handler)
    queue_listener.start()

    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger("logger_sits")
    logger.setLevel(logging.ERROR)
    logger.addHandler(queue_handler)
    logger.propagate = False

    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    ee_feature = ee.Feature(
        ee_geometry,
        {"s": start_date, "e": end_date, "0": 1},
    )
    ee_expression = satellite.imageCollection(ee_feature)

    if ee_expression.size().getInfo() == 0:
        logger.error("No images found for the specified parameters.")
        return np.array([]), []

    max_valid_pixels = ee_expression.aggregate_max("ZZ_USER_VALID_PIXELS")
    threshold = ee.Number(max_valid_pixels).multiply(invalid_images_threshold)
    ee_expression = ee_expression.filter(ee.Filter.gte("ZZ_USER_VALID_PIXELS", threshold))

    image_names = ee_expression.aggregate_array("ZZ_USER_TIME_DUMMY").getInfo()

    image_indexes = [
        (n, image_index)
        for n, image_index in enumerate(ee.data.computeValue(ee_expression.aggregate_array("system:index")))
    ]
    image_indexes_with_errors = []
    all_images = [np.array([]) for _ in range(len(image_indexes))]

    def process_download(i: int, image_index: str) -> tuple[np.ndarray, int, str]:
        try:
            img = ee_img_to_numpy(
                ee_expression.filter(ee.Filter.eq("system:index", image_index)).first().clip(ee_geometry),
                ee_geometry,
                satellite.pixelSize,
            )
            return img, i, image_index  # noqa: TRY300
        except Exception as e:
            logger.error(f"download_multiple_images_multithread_{i}_{satellite.shortName} = {e}")  # noqa: TRY400
            return np.array([]), i, image_index

    def run_downloads(image_indexes: list[tuple[int, str]], num_threads: int, desc: str) -> None:
        nonlocal all_images
        error_count = 0

        with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
            futures = [executor.submit(partial(process_download, n, image_index)) for n, image_index in image_indexes]

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc)
            for future in pbar:
                result_img, i, image_index = future.result()
                if result_img.shape[0] == 0:
                    error_count += 1
                    image_indexes_with_errors.append((i, image_index))
                    pbar.set_postfix({"errors": error_count})
                else:
                    all_images[i] = result_img

    run_downloads(image_indexes, num_threads=num_threads_rush, desc="Downloading")

    if image_indexes_with_errors:
        run_downloads(
            image_indexes_with_errors,
            num_threads=num_threads_retry,
            desc="Re-running failed downloads",
        )

    return np.stack(all_images), image_names


def download_single_image(
    geometry: Polygon,
    satellite: SingleImageSatellite,
) -> np.ndarray:
    log_queue: queue.Queue[logging.LogRecord] = queue.Queue(-1)

    file_handler = logging.FileHandler("logging.log", mode="a")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    queue_listener = logging.handlers.QueueListener(log_queue, file_handler)
    queue_listener.start()

    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger("logger_sits")
    logger.setLevel(logging.ERROR)
    logger.addHandler(queue_handler)
    logger.propagate = False

    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    ee_feature = ee.Feature(ee_geometry, {"0": 1})

    try:
        image = satellite.image(ee_feature)
        image_clipped = image.clip(ee_geometry)
        image_np = ee_img_to_numpy(image_clipped, ee_geometry, satellite.pixelSize)
    except Exception as e:
        logger.exception(f"download_single_image_{satellite.shortName} = {e}")  # noqa: TRY401
        return np.array([])

    return image_np
