import asyncio
import logging
import pathlib
from typing import Any, cast

import aiohttp
import ee
import numpy as np
import pandas as pd
from shapely import MultiPolygon, Point, Polygon
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential
from tqdm.std import tqdm

from agrigee_lite.config import (
    AIOHTTP_CONNECTOR_LIMIT,
    AIOHTTP_TIMEOUT_SECONDS,
    ASYNC_MAX_PARALLEL_DOWNLOADS,
    ASYNC_MAX_RETRIES_PER_CHUNK,
)
from agrigee_lite.ee_utils import ee_img_to_numpy
from agrigee_lite.misc import create_dict_hash, log_dict_function_call_summary
from agrigee_lite.sat.abstract_satellite import AbstractSatellite, SingleImageSatellite


def _as_date_str(value: pd.Timestamp | str) -> str:
    return str(value)[:10]


def download_multiple_images(
    geometry: Polygon | MultiPolygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    invalid_images_threshold: float = 0.5,
    max_parallel_downloads: int = ASYNC_MAX_PARALLEL_DOWNLOADS,
    force_redownload: bool = False,
    image_indices: list[int] | None = None,
    max_retries_per_chunk: int = ASYNC_MAX_RETRIES_PER_CHUNK,
) -> list[str]:
    """Download raw satellite images (as GeoTIFF ZIPs) for a geometry and date range.

    Each image is saved as a ``.zip`` file in
    ``~/.cache/agrigee_lite/images/<hash>/``, where the hash encodes all
    relevant parameters so cached results are reused automatically on
    subsequent calls.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon
        Area of interest.
    start_date : pd.Timestamp or str
        Start of the date range (ISO-8601 or ``YYYY-MM-DD``).
    end_date : pd.Timestamp or str
        End of the date range (inclusive).
    satellite : AbstractSatellite
        Satellite configuration, e.g. ``Sentinel2(bands={"red", "nir"})``.
    invalid_images_threshold : float, default 0.5
        Fraction of the maximum valid-pixel count used as a quality filter.
        Images with fewer valid pixels than
        ``max_valid_pixels * invalid_images_threshold`` are excluded.
        Set to 0.0 to keep all images regardless of cloud cover.
    max_parallel_downloads : int, default 40
        Maximum simultaneous downloads.
    force_redownload : bool, default False
        Delete cached ZIPs and re-download from scratch.
    image_indices : list of int or None, optional
        If given, only download the images at these positions in the
        (filtered, sorted) collection.  Useful for previewing a subset
        without downloading everything.
    max_retries_per_chunk : int, default 5
        Maximum retry attempts per image download.

    Returns
    -------
    list of str
        Dates of the downloaded images in ``YYYY-MM-DD`` format, in the same
        order as the files on disk.
    """

    start_date_str = _as_date_str(start_date)
    end_date_str = _as_date_str(end_date)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            download_multiple_images_async(
                geometry=geometry,
                start_date=start_date_str,
                end_date=end_date_str,
                satellite=satellite,
                invalid_images_threshold=invalid_images_threshold,
                max_parallel_downloads=max_parallel_downloads,
                force_redownload=force_redownload,
                image_indices=image_indices,
                max_retries_per_chunk=max_retries_per_chunk,
            )
        )

    raise RuntimeError(
        "download_multiple_images cannot run inside an active event loop. Use download_multiple_images_async instead."
    )


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


async def _resolve_ee_image_collection(
    ee_expression: ee.ImageCollection,
    invalid_images_threshold: float,
    image_indices: list[int] | None,
) -> tuple[list[str], list[str]]:
    """Fetch image names and indexes from EE, apply threshold and index filters."""
    max_valid_pixels = ee_expression.aggregate_max("ZZ_USER_VALID_PIXELS")
    threshold = ee.Number(max_valid_pixels).multiply(invalid_images_threshold)
    ee_expression = ee_expression.filter(ee.Filter.gte("ZZ_USER_VALID_PIXELS", threshold))

    _gathered = await asyncio.gather(
        asyncio.to_thread(ee_expression.aggregate_array("ZZ_USER_TIME_DUMMY").getInfo),
        asyncio.to_thread(ee_expression.aggregate_array("system:index").getInfo),
    )
    image_names: list[str] = cast(list[str], _gathered[0])
    image_indexes: list[str] = cast(list[str], _gathered[1])

    if image_indices is not None:
        valid_indices = [i for i in image_indices if 0 <= i < len(image_indexes)]
        if not valid_indices:
            return [], []
        image_names = [image_names[i] for i in valid_indices]
        image_indexes = [image_indexes[i] for i in valid_indices]

    return image_names, image_indexes


async def _download_url_to_path(
    session: aiohttp.ClientSession,
    url: str,
    output_path: pathlib.Path,
) -> None:
    timeout = aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT_SECONDS)
    async with session.get(url, timeout=timeout) as response:
        response.raise_for_status()
        content = await response.read()
    output_path.write_bytes(content)


async def _fetch_and_download_image(
    chunk_index: int,
    ee_expression: ee.ImageCollection,
    image_names: list[str],
    image_indexes: list[str],
    ee_geometry: ee.Geometry,
    session: aiohttp.ClientSession,
    output_dir: pathlib.Path,
    semaphore: asyncio.Semaphore,
    max_retries_per_chunk: int,
) -> tuple[int, bool]:
    """Resolve a single GEE download URL and save its ZIP payload to disk."""
    async with semaphore:
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(max_retries_per_chunk),
                wait=wait_exponential(multiplier=1, min=1, max=30),
            ):
                with attempt:
                    img = ee.Image(
                        ee_expression.filter(ee.Filter.eq("system:index", image_indexes[chunk_index])).first()
                    )
                    url = await asyncio.wait_for(
                        asyncio.to_thread(
                            img.getDownloadURL, {"name": image_names[chunk_index], "region": ee_geometry}
                        ),
                        timeout=180,
                    )
                    file_path = output_dir / f"{image_names[chunk_index]}.zip"
                    await _download_url_to_path(session, url, file_path)
            return chunk_index, True  # noqa: TRY300
        except RetryError:
            logging.exception("Image chunk %d failed after %d attempts.", chunk_index, max_retries_per_chunk)
            return chunk_index, False
        except Exception:
            logging.exception("Image chunk %d failed with unexpected error.", chunk_index)
            return chunk_index, False


async def download_multiple_images_async(
    geometry: Polygon | MultiPolygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    invalid_images_threshold: float = 0.5,
    max_parallel_downloads: int = ASYNC_MAX_PARALLEL_DOWNLOADS,
    force_redownload: bool = False,
    image_indices: list[int] | None = None,
    max_retries_per_chunk: int = ASYNC_MAX_RETRIES_PER_CHUNK,
) -> list[str]:
    """Async version of :func:`download_multiple_images`.

    Identical semantics to the synchronous version but non-blocking: GEE URL
    resolution runs in a thread pool via ``asyncio.to_thread`` and payload
    downloads run through ``aiohttp``, so the event loop stays responsive. Use
    this variant inside the REST API or any other async context.

    Parameters
    ----------
    geometry : Polygon or MultiPolygon
        Area of interest.
    start_date : pd.Timestamp or str
        Start of the date range.
    end_date : pd.Timestamp or str
        End of the date range.
    satellite : AbstractSatellite
        Satellite configuration.
    invalid_images_threshold : float, default 0.5
        Quality filter — see :func:`download_multiple_images`.
    max_parallel_downloads : int, default 40
        Maximum simultaneous downloads.
    force_redownload : bool, default False
        Re-download even if cached ZIPs exist.
    image_indices : list of int or None, optional
        Restrict to specific collection positions.
    max_retries_per_chunk : int, default 5
        Maximum retry attempts per image download.

    Returns
    -------
    list of str
        Dates of the downloaded images in ``YYYY-MM-DD`` format.
    """
    start_date = _as_date_str(start_date)
    end_date = _as_date_str(end_date)

    ee_geometry = ee.Geometry(geometry.__geo_interface__)
    ee_feature = ee.Feature(ee_geometry, {"s": start_date, "e": end_date, "0": 1})
    ee_expression = satellite.imageCollection(ee_feature)

    metadata_dict: dict[str, Any] = {}
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

    collection_size = await asyncio.to_thread(ee_expression.size().getInfo)
    if collection_size == 0:
        print("No images found for the specified parameters.")
        return []
    print(f"Found {collection_size} images for the specified parameters.")

    image_names, image_indexes = await _resolve_ee_image_collection(
        ee_expression, invalid_images_threshold, image_indices
    )
    if not image_names:
        print("No valid image indices provided.")
        return []

    output_path = pathlib.Path.home() / ".cache" / "agrigee_lite" / "images" / f"{create_dict_hash(metadata_dict)}"
    output_path.mkdir(parents=True, exist_ok=True)

    if force_redownload:
        for f in output_path.glob("*.zip"):
            f.unlink()

    already_downloaded_stems = {x.stem for x in output_path.glob("*.zip")}
    pending_chunks = sorted(i for i in range(len(image_indexes)) if image_names[i] not in already_downloaded_stems)

    pbar = tqdm(total=len(pending_chunks), desc=f"Downloading images ({output_path.name})", unit="feature")
    if not pending_chunks:
        pbar.close()
        return image_names

    failed_chunks: list[int] = []
    semaphore = asyncio.Semaphore(max_parallel_downloads)
    connector = aiohttp.TCPConnector(limit=max(max_parallel_downloads, AIOHTTP_CONNECTOR_LIMIT))
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                _fetch_and_download_image(
                    chunk_index=i,
                    ee_expression=ee_expression,
                    image_names=image_names,
                    image_indexes=image_indexes,
                    ee_geometry=ee_geometry,
                    session=session,
                    output_dir=output_path,
                    semaphore=semaphore,
                    max_retries_per_chunk=max_retries_per_chunk,
                )
            )
            for i in pending_chunks
        ]

        for task in asyncio.as_completed(tasks):
            chunk_id, success = await task
            if not success:
                failed_chunks.append(chunk_id)
            pbar.update(1)

    pbar.close()

    if failed_chunks:
        raise RuntimeError(f"Failed to download {len(failed_chunks)} image(s): {sorted(failed_chunks)}")

    return image_names
