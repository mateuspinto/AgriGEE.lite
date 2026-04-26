import asyncio
import getpass
import io
import json
import logging
import signal
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any

import aiohttp
import ee
import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
from shapely import MultiPolygon, Point, Polygon
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_exponential
from tqdm.std import tqdm

from agrigee_lite.cache.spatialite_cache import (
    compute_geom_hash,
    fetch_sits,
    fetch_sits_by_request_ids,
    fetch_sits_cache_index,
    get_engine,
    store_sits,
)
from agrigee_lite.ee_utils import (
    ee_gdf_to_feature_collection,
    ee_get_tasks_status,
)
from agrigee_lite.misc import (
    create_gdf_hash,
    get_reducer_names,
    h3_clustering,
    log_dict_function_call_summary,
)
from agrigee_lite.sat.abstract_satellite import AbstractSatellite, OpticalSatellite
from agrigee_lite.task_manager import GEETaskManager


def build_ee_expression(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    original_index_column_name: str,
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
) -> ee.FeatureCollection:
    """
    Build Earth Engine expression for satellite time series computation.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing geometries and date information.
    satellite : AbstractSatellite
        Satellite configuration object.
    reducers : set[str] or None
        Set of reducer names to apply to the computation.
    subsampling_max_pixels : float
        Maximum pixels for sampling: >1 = absolute count, ≤1 = fraction of area (e.g., 0.5 = 50% sampling).
    original_index_column_name : str
        Name of the column containing original indices.
    start_date_column_name : str, optional
        Name of the start date column, by default "start_date".
    end_date_column_name : str, optional
        Name of the end date column, by default "end_date".

    Returns
    -------
    ee.FeatureCollection
        Earth Engine FeatureCollection with computed satellite time series.
    """
    fc = ee_gdf_to_feature_collection(gdf, original_index_column_name, start_date_column_name, end_date_column_name)
    return ee.FeatureCollection(
        fc.map(
            partial(
                satellite.compute,
                reducers=reducers,
                subsampling_max_pixels=subsampling_max_pixels,
            )
        )
    ).flatten()


def build_selectors(satellite: AbstractSatellite, reducers: set[str] | None) -> list[str]:
    """Return the GEE column selectors that map to the output DataFrame columns.

    Parameters
    ----------
    satellite : AbstractSatellite
        Configured satellite whose selected bands/indices define the columns.
    reducers : set of str or None
        When ``None`` or a single reducer, columns are the raw band names.
        When multiple reducers are given, a ``<band>_<reducer>`` column is
        produced for every combination.

    Returns
    -------
    list of str
        Ordered list of property names to extract from the GEE FeatureCollection.
    """
    if (reducers is None) or (len(reducers) == 1):
        return [
            "00_indexnum",
            "01_timestamp",
            *satellite.toDownloadSelectors,
            "99_validPixelsCount",
        ]

    else:
        reducer_names = get_reducer_names(reducers)
        return [
            "00_indexnum",
            "01_timestamp",
            *[
                f"{numeral_band_name}_{reducer_name}"
                for _, numeral_band_name in satellite.selectedBands
                for reducer_name in reducer_names
            ],
            *[
                f"{numeral_indice_name}_{reducer_name}"
                for _, _, numeral_indice_name in satellite.selectedIndices
                for reducer_name in reducer_names
            ],
            "99_validPixelsCount",
        ]


def prepare_output_df(df: pd.DataFrame, satellite: AbstractSatellite, original_index_column_name: str) -> pd.DataFrame:
    """
    Prepare and clean output DataFrame from satellite time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame from satellite time series computation.
    satellite : AbstractSatellite
        Satellite configuration object used for data processing.
    original_index_column_name : str
        Name of the column to restore original indices.

    Returns
    -------
    pd.DataFrame
        Cleaned and processed DataFrame with proper column names and data types.
    """
    df = df.copy()

    df = df.drop(columns=["geo"], errors="ignore")
    df.columns = [column.split("_", 1)[1] if "_" in column else column for column in df.columns.tolist()]

    if isinstance(satellite, OpticalSatellite):  # Zero values in optical bands are invalid
        band_columns = sorted(set(df.columns) - {"timestamp", "validPixelsCount", "indexnum"})
        df = df[~(df[band_columns] == 0).all(axis=1)]

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d")

    if "timestamp" in df.columns and df["timestamp"].isna().all():
        df = df.drop(columns=["timestamp"])

    if "indexnum" in df.columns and (df["indexnum"] == 0).all():
        df = df.drop(columns=["indexnum"])
    elif "indexnum" in df.columns:
        df = df.sort_values(by=["indexnum"], kind="stable")
        df = df.reset_index(drop=True)

    if "indexnum" in df.columns:
        df = df.rename(columns={"indexnum": original_index_column_name})

    return df


def sanitize_and_prepare_input_gdf(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    original_index_column_name: str,
    coarse_resolution: int = 5,
    fine_resolution: int = 8,
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
) -> gpd.GeoDataFrame:
    """
    Sanitize and prepare input GeoDataFrame for satellite time series processing.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame with geometries and temporal information.
    satellite : AbstractSatellite
        Satellite configuration object.
    original_index_column_name : str
        Name of the column to store original indices.
    coarse_resolution : int, optional
        H3 resolution for cluster boundaries (default 5).
    fine_resolution : int, optional
        H3 resolution for intra-cluster ordering (default 8).
    start_date_column_name : str, optional
        Name of the start date column, by default "start_date".
    end_date_column_name : str, optional
        Name of the end date column, by default "end_date".

    Returns
    -------
    gpd.GeoDataFrame
        Sanitized GeoDataFrame with clustering applied and invalid data filtered.
    """
    gdf = gdf.copy()

    if original_index_column_name == "original_index":
        gdf = gdf.reset_index().rename(columns={"index": original_index_column_name})
        logging.info(f"Column '{original_index_column_name}' created to store original index.")

    schema = pa.DataFrameSchema(
        {
            "geometry": pa.Column("geometry", nullable=False),
            start_date_column_name: pa.Column(pa.DateTime, nullable=False),
            end_date_column_name: pa.Column(pa.DateTime, nullable=False),
            original_index_column_name: pa.Column(gdf[original_index_column_name].dtype),
        },
        unique=[original_index_column_name],
    )
    schema.validate(gdf, lazy=True)

    gdf = gdf[["geometry", start_date_column_name, end_date_column_name, original_index_column_name]]

    mask_no_intersection = (gdf[end_date_column_name] < satellite.startDate) | (
        gdf[start_date_column_name] > satellite.endDate
    )
    mask_total_intersection = (gdf[start_date_column_name] >= satellite.startDate) & (
        gdf[end_date_column_name] <= satellite.endDate
    )
    mask_partial_intersection = ~(mask_no_intersection | mask_total_intersection)

    count_none = mask_no_intersection.sum()
    count_partial = mask_partial_intersection.sum()

    pct_none = 100 * count_none / len(gdf)
    if pct_none > 0:
        logging.warning(f"{pct_none:.2f}% of the data do not intersect the satellite period.")

    pct_partial = 100 * count_partial / len(gdf)
    if pct_partial > 0:
        logging.info(f"{pct_partial:.2f}% of the data partially intersect the satellite period.")

    if pct_none == 100:
        return gpd.GeoDataFrame()

    gdf = gdf[~mask_no_intersection].reset_index(drop=True)

    gdf = h3_clustering(gdf, coarse_resolution=coarse_resolution, fine_resolution=fine_resolution)

    return gdf


def download_single_sits(
    geometry: Polygon | MultiPolygon | Point,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    reducers: set[str] | None = None,
    subsampling_max_pixels: float = 1_000,
) -> pd.DataFrame:
    """
    Download satellite time series for a single geometry.

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, or Point
        The area or point of interest for data extraction.
    start_date : pd.Timestamp or str
        Start date for time series collection.
    end_date : pd.Timestamp or str
        End date for time series collection.
    satellite : AbstractSatellite
        Satellite configuration object.
    reducers : set[str] or None, optional
        Set of reducer names to apply, by default None.
    subsampling_max_pixels : float, optional
        Maximum pixels for sampling: >1 = absolute count, ≤1 = fraction of area (e.g., 0.5 = 50% sampling), by default 1_000.

    Returns
    -------
    pd.DataFrame
        DataFrame containing satellite time series data.

    Raises
    ------
    ValueError
        If the requested period does not intersect with satellite's temporal range.
    """
    start_date = start_date.strftime("%Y-%m-%d") if isinstance(start_date, pd.Timestamp) else start_date
    end_date = end_date.strftime("%Y-%m-%d") if isinstance(end_date, pd.Timestamp) else end_date

    if end_date < satellite.startDate or start_date > satellite.endDate:
        raise ValueError(  # noqa: TRY003
            f"Requested period ({start_date} to {end_date}) does not intersect with satellite's range "
            f"({satellite.startDate} to {satellite.endDate})"
        )

    _engine = get_engine()
    if _engine is not None:
        cached = fetch_sits(_engine, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)
        if cached is not None:
            logging.debug("Cache hit: %s %s→%s", satellite.shortName, start_date, end_date)
            return cached

    ee_feature = ee.Feature(
        geometry.__geo_interface__,
        {"s": start_date, "e": end_date, "0": 0},
    )
    ee_expression = satellite.compute(ee_feature, reducers=reducers, subsampling_max_pixels=subsampling_max_pixels)

    sits_df = ee.data.computeFeatures({"expression": ee_expression, "fileFormat": "PANDAS_DATAFRAME"})
    sits_df = prepare_output_df(sits_df, satellite, "IGNORED")

    if _engine is not None:
        store_sits(_engine, sits_df, geometry, start_date, end_date, satellite, reducers, subsampling_max_pixels)

    return sits_df


async def download_multiple_sits_async(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None = None,
    original_index_column_name: str = "original_index",
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
    subsampling_max_pixels: float = 1_000,
    chunksize: int = 10,
    max_concurrent: int = 20,
    max_retries: int = 3,
    force_redownload: bool = False,
) -> pd.DataFrame:
    if len(gdf) == 0:
        return pd.DataFrame()

    gdf = sanitize_and_prepare_input_gdf(
        gdf,
        satellite,
        original_index_column_name,
        start_date_column_name=start_date_column_name,
        end_date_column_name=end_date_column_name,
    )

    if len(gdf) == 0:
        return pd.DataFrame()

    _engine = get_engine()
    cached_frames: list[pd.DataFrame] = []
    if _engine is not None and not force_redownload:
        cache_index = fetch_sits_cache_index(_engine, satellite, reducers, subsampling_max_pixels)
        if cache_index:
            hit_request_ids: dict[int, Any] = {}
            uncached_positions: list[Any] = []
            for pos, row in gdf.iterrows():
                key = (
                    compute_geom_hash(row.geometry),
                    str(row[start_date_column_name])[:10],
                    str(row[end_date_column_name])[:10],
                )
                req_id = cache_index.get(key)
                if req_id is not None:
                    hit_request_ids[req_id] = (pos, row[original_index_column_name])
                else:
                    uncached_positions.append(pos)

            if hit_request_ids:
                cached_data = fetch_sits_by_request_ids(_engine, satellite, list(hit_request_ids.keys()))
                for req_id, df in cached_data.items():
                    _, orig_idx = hit_request_ids[req_id]
                    df[original_index_column_name] = orig_idx
                    cached_frames.append(df)

            gdf = gdf.loc[uncached_positions].reset_index(drop=True)
            if gdf.empty:
                return pd.concat(cached_frames, ignore_index=True) if cached_frames else pd.DataFrame()

    num_chunks = (len(gdf) + chunksize - 1) // chunksize
    selectors = build_selectors(satellite, reducers)
    semaphore = asyncio.Semaphore(max_concurrent)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="agrigee_gee")
    pbar = tqdm(
        total=num_chunks,
        unit="chunk",
        smoothing=0,
        bar_format="{percentage:3.0f}% | {n_fmt}/{total_fmt} | [{elapsed}<{remaining}, {rate_fmt}]",
    )

    async def fetch_chunk(session: aiohttp.ClientSession, chunk_id: int) -> pd.DataFrame:
        sub = gdf.iloc[chunk_id * chunksize : (chunk_id + 1) * chunksize].copy()

        def get_url() -> str:
            expr = build_ee_expression(
                sub,
                satellite,
                reducers,
                subsampling_max_pixels,
                original_index_column_name,
                start_date_column_name,
                end_date_column_name,
            )
            return expr.getDownloadURL(filetype="csv", selectors=selectors, filename=str(chunk_id))

        url = await loop.run_in_executor(executor, get_url)

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=600)) as resp:
            resp.raise_for_status()
            data = await resp.read()

        return pd.read_csv(io.BytesIO(data))

    async def fetch_with_retry(session: aiohttp.ClientSession, chunk_id: int) -> pd.DataFrame | None:
        async with semaphore:
            try:
                _result: pd.DataFrame | None = None
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_retries),
                    wait=wait_exponential(multiplier=1, min=2, max=30),
                ):
                    with attempt:
                        _result = await fetch_chunk(session, chunk_id)
                return _result
            except RetryError:
                logging.error("Chunk %d failed after %d attempts.", chunk_id, max_retries)
                return None
            finally:
                pbar.update(1)

    def _cancel_all() -> None:
        import os
        logging.warning("Download interrupted. Exiting.")
        os._exit(1)

    _signals_registered = False
    try:
        loop.add_signal_handler(signal.SIGINT, _cancel_all)
        loop.add_signal_handler(signal.SIGTERM, _cancel_all)
        _signals_registered = True
    except (NotImplementedError, ValueError):
        pass

    connector = aiohttp.TCPConnector(limit=max_concurrent)
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [asyncio.create_task(fetch_with_retry(session, cid)) for cid in range(num_chunks)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        if _signals_registered:
            loop.remove_signal_handler(signal.SIGINT)
            loop.remove_signal_handler(signal.SIGTERM)
        pbar.close()

    frames: list[pd.DataFrame] = []
    for cid, res in enumerate(results):
        if isinstance(res, BaseException):
            logging.error("Chunk %d raised: %s", cid, res)
        elif res is not None and not res.empty:
            frames.append(res)

    whole_result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    whole_result_df = prepare_output_df(whole_result_df, satellite, original_index_column_name)

    if _engine is not None and not whole_result_df.empty:
        for orig_idx, feature_df in whole_result_df.groupby(original_index_column_name, sort=False):
            orig_row = gdf[gdf[original_index_column_name] == orig_idx].iloc[0]
            store_sits(
                _engine,
                feature_df.drop(columns=[original_index_column_name]),
                orig_row.geometry,
                str(orig_row[start_date_column_name])[:10],
                str(orig_row[end_date_column_name])[:10],
                satellite,
                reducers,
                subsampling_max_pixels,
            )

    if cached_frames:
        whole_result_df = pd.concat([pd.concat(cached_frames, ignore_index=True), whole_result_df], ignore_index=True)

    return whole_result_df


def download_multiple_sits_chunks_gdrive(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None = None,
    original_index_column_name: str = "original_index",
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
    subsampling_max_pixels: float = 1_000,
    coarse_resolution: int = 5,
    fine_resolution: int = 8,
    gee_save_folder: str = "AGL_EXPORTS",
    force_redownload: bool = False,
    wait: bool = True,
) -> None:
    """
    Download satellite time series using Google Earth Engine tasks to Google Drive.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing geometries and temporal information.
    satellite : AbstractSatellite
        Satellite configuration object.
    reducers : set[str] or None, optional
        Set of reducer names to apply, by default None.
    original_index_column_name : str, optional
        Name of the column to store original indices, by default "original_index".
    start_date_column_name : str, optional
        Name of the start date column, by default "start_date".
    end_date_column_name : str, optional
        Name of the end date column, by default "end_date".
    subsampling_max_pixels : float, optional
        Maximum pixels for sampling: >1 = absolute count, ≤1 = fraction of area (e.g., 0.5 = 50% sampling), by default 1_000.
    coarse_resolution : int, optional
        H3 resolution for cluster boundaries, by default 5.
    fine_resolution : int, optional
        H3 resolution for intra-cluster ordering, by default 8.
    gee_save_folder : str, optional
        Google Drive folder name for saving exports, by default "AGL_EXPORTS".
    force_redownload : bool, optional
        Whether to force re-download of existing data, by default False.
    wait : bool, optional
        Whether to wait for task completion, by default True.
    """
    if len(gdf) == 0:
        return None

    def download_multiple_sits_task_gdrive(
        gdf: gpd.GeoDataFrame,
        satellite: AbstractSatellite,
        file_stem: str,
        reducers: set[str] | None = None,
        original_index_column_name: str = "original_index",
        start_date_column_name: str = "start_date",
        end_date_column_name: str = "end_date",
        subsampling_max_pixels: float = 1_000,
        taskname: str = "",
        gee_save_folder: str = "AGL_EXPORTS",
    ) -> ee.batch.Task:
        """
        Create a Google Earth Engine export task to Google Drive.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing geometries and temporal information.
        satellite : AbstractSatellite
            Satellite configuration object.
        file_stem : str
            Base filename for the exported file.
        reducers : set[str] or None, optional
            Set of reducer names to apply, by default None.
        original_index_column_name : str, optional
            Name of the column to store original indices, by default "original_index".
        start_date_column_name : str, optional
            Name of the start date column, by default "start_date".
        end_date_column_name : str, optional
            Name of the end date column, by default "end_date".
        subsampling_max_pixels : float, optional
            Maximum pixels for sampling: >1 = absolute count, ≤1 = fraction of area (e.g., 0.5 = 50% sampling), by default 1_000.
        taskname : str, optional
            Custom task name, by default "".
        gee_save_folder : str, optional
            Google Drive folder name, by default "AGL_EXPORTS".

        Returns
        -------
        ee.batch.Task
            Earth Engine export task object.
        """
        if taskname == "":
            taskname = file_stem

        ee_expression = build_ee_expression(
            gdf,
            satellite,
            reducers,
            subsampling_max_pixels,
            original_index_column_name,
            start_date_column_name,
            end_date_column_name,
        )

        task = ee.batch.Export.table.toDrive(
            collection=ee_expression,
            description=taskname,
            fileFormat="CSV",
            fileNamePrefix=file_stem,
            folder=gee_save_folder,
            selectors=build_selectors(satellite, reducers),
        )

        return task

    gdf = sanitize_and_prepare_input_gdf(
        gdf,
        satellite,
        original_index_column_name,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
        start_date_column_name=start_date_column_name,
        end_date_column_name=end_date_column_name,
    )

    task_mgr = GEETaskManager()

    tasks_df = ee_get_tasks_status()
    tasks_df = tasks_df[tasks_df.description.str.startswith("agl_multiple_sits_")]
    completed_or_running_tasks = set(
        tasks_df.description.apply(lambda x: x.split("_", 1)[0] + "_" + x.split("_", 2)[2]).tolist()
    )  # The task is the same, no matter who started it

    username = getpass.getuser().replace("_", "")
    hashname = create_gdf_hash(gdf, start_date_column_name, end_date_column_name)

    for cluster_id in tqdm(
        sorted(gdf.cluster_id.unique()),
        desc=f"Creating GEE tasks ({satellite.shortName}_{hashname}_r{coarse_resolution})",
    ):
        cluster_id = int(cluster_id)

        if (force_redownload) or (
            f"agl_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}" not in completed_or_running_tasks
        ):
            task = download_multiple_sits_task_gdrive(
                gdf[gdf.cluster_id == cluster_id],
                satellite,
                f"{satellite.shortName}_{hashname}_{cluster_id}",
                reducers=reducers,
                original_index_column_name=original_index_column_name,
                start_date_column_name=start_date_column_name,
                end_date_column_name=end_date_column_name,
                subsampling_max_pixels=subsampling_max_pixels,
                taskname=f"agl_{username}_sits_{satellite.shortName}_{hashname}_{cluster_id}",
                gee_save_folder=gee_save_folder,
            )

            task_mgr.add(task)

    task_mgr.start()  # Start all tasks at once allows user to cancel them before submitted to GEE

    if wait:
        task_mgr.wait()


def download_multiple_sits_chunks_gcs(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    bucket_name: str,
    reducers: set[str] | None = None,
    original_index_column_name: str = "original_index",
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
    subsampling_max_pixels: float = 1_000,
    coarse_resolution: int = 5,
    fine_resolution: int = 8,
    force_redownload: bool = False,
    wait: bool = True,
) -> None | pd.DataFrame:
    """
    Download satellite time series using Google Earth Engine tasks to Google Cloud Storage.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing geometries and temporal information.
    satellite : AbstractSatellite
        Satellite configuration object.
    bucket_name : str
        Google Cloud Storage bucket name for exports.
    reducers : set[str] or None, optional
        Set of reducer names to apply, by default None.
    original_index_column_name : str, optional
        Name of the column to store original indices, by default "original_index".
    start_date_column_name : str, optional
        Name of the start date column, by default "start_date".
    end_date_column_name : str, optional
        Name of the end date column, by default "end_date".
    subsampling_max_pixels : float, optional
        Maximum pixels for sampling: >1 = absolute count, ≤1 = fraction of area (e.g., 0.5 = 50% sampling), by default 1_000.
    coarse_resolution : int, optional
        H3 resolution for cluster boundaries, by default 5.
    fine_resolution : int, optional
        H3 resolution for intra-cluster ordering, by default 8.
    force_redownload : bool, optional
        Whether to force re-download of existing data, by default False.
    wait : bool, optional
        Whether to wait for task completion, by default True.

    Returns
    -------
    None or pd.DataFrame
        If wait is True, returns DataFrame with combined results.
        If wait is False, returns None.
    """
    from smart_open import open  # pyright: ignore[reportMissingImports]

    if len(gdf) == 0:
        logging.warning("Empty GeoDataFrame, nothing to download")
        return None

    def download_multiple_sits_task_gcs(
        gdf: gpd.GeoDataFrame,
        satellite: AbstractSatellite,
        bucket_name: str,
        file_path: str,
        reducers: set[str] | None = None,
        original_index_column_name: str = "original_index",
        start_date_column_name: str = "start_date",
        end_date_column_name: str = "end_date",
        subsampling_max_pixels: float = 1_000,
        taskname: str = "",
    ) -> ee.batch.Task:
        """
        Create a Google Earth Engine export task to Google Cloud Storage.

        Parameters
        ----------
        gdf : gpd.GeoDataFrame
            GeoDataFrame containing geometries and temporal information.
        satellite : AbstractSatellite
            Satellite configuration object.
        bucket_name : str
            Google Cloud Storage bucket name.
        file_path : str
            File path within the bucket for the exported file.
        reducers : set[str] or None, optional
            Set of reducer names to apply, by default None.
        original_index_column_name : str, optional
            Name of the column to store original indices, by default "original_index".
        start_date_column_name : str, optional
            Name of the start date column, by default "start_date".
        end_date_column_name : str, optional
            Name of the end date column, by default "end_date".
        subsampling_max_pixels : float, optional
            Maximum pixels for sampling: >1 = absolute count, ≤1 = fraction of area (e.g., 0.5 = 50% sampling), by default 1_000.
        taskname : str, optional
            Custom task name, by default "".

        Returns
        -------
        ee.batch.Task
            Earth Engine export task object.
        """
        if taskname == "":
            taskname = file_path

        ee_expression = build_ee_expression(
            gdf,
            satellite,
            reducers,
            subsampling_max_pixels,
            original_index_column_name,
            start_date_column_name,
            end_date_column_name,
        )

        task = ee.batch.Export.table.toCloudStorage(
            bucket=bucket_name,
            collection=ee_expression,
            description=taskname,
            fileFormat="CSV",
            fileNamePrefix=file_path,
            selectors=build_selectors(satellite, reducers),
        )

        return task

    gdf = sanitize_and_prepare_input_gdf(
        gdf,
        satellite,
        original_index_column_name,
        coarse_resolution=coarse_resolution,
        fine_resolution=fine_resolution,
        start_date_column_name=start_date_column_name,
        end_date_column_name=end_date_column_name,
    )

    task_mgr = GEETaskManager()
    tasks_df = ee_get_tasks_status()
    tasks_df = tasks_df[tasks_df.description.str.startswith("agl_")].reset_index(drop=True)
    completed_or_running_tasks = set(
        tasks_df.description.apply(lambda x: x.split("_", 1)[0] + "_" + x.split("_", 2)[2]).tolist()
    )  # The task is the same, no matter who started it

    username = getpass.getuser().replace("_", "")
    hashname = create_gdf_hash(gdf, start_date_column_name, end_date_column_name)

    gcs_save_folder = f"agl/{satellite.shortName}_{hashname}"
    metadata_dict: dict[str, Any] = {}
    metadata_dict |= log_dict_function_call_summary(["gdf", "satellite"])
    metadata_dict |= satellite.log_dict()
    metadata_dict["user"] = username
    metadata_dict["creation_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(f"gs://{bucket_name}/{gcs_save_folder}/metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=4)

    with open(f"gs://{bucket_name}/{gcs_save_folder}/geodataframe.parquet", "wb") as f:
        gdf.to_parquet(f, compression="brotli")

    file_uris = []

    for cluster_id in tqdm(sorted(gdf.cluster_id.unique())):
        cluster_id = int(cluster_id)

        if (force_redownload) or (
            f"agl_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}" not in completed_or_running_tasks
        ):
            # TODO: Also skip if the file already exists in GCS
            task = download_multiple_sits_task_gcs(
                gdf[gdf.cluster_id == cluster_id],
                satellite,
                bucket_name=bucket_name,
                file_path=f"{gcs_save_folder}/{cluster_id}",
                reducers=reducers,
                original_index_column_name=original_index_column_name,
                start_date_column_name=start_date_column_name,
                end_date_column_name=end_date_column_name,
                subsampling_max_pixels=subsampling_max_pixels,
                taskname=f"agl_{username}_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}",
            )

            task_mgr.add(task)

        file_uris.append(f"gs://{bucket_name}/{gcs_save_folder}/{cluster_id}.csv")

    task_mgr.start()

    if wait:
        task_mgr.wait()

        df = pd.DataFrame()
        for file_uri in file_uris:
            with open(file_uri, "r") as f:
                sub_df = pd.read_csv(f)
                df = pd.concat([df, sub_df], ignore_index=True)

        df = prepare_output_df(df, satellite, original_index_column_name)
        return df
    else:
        return None
