import asyncio
import getpass
import json
import logging
import pathlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any

import ee
import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
from shapely import MultiPolygon, Point, Polygon
from tqdm.std import tqdm

from agrigee_lite.cache.spatialite_cache import fetch_sits, get_engine, store_sits
from agrigee_lite.downloader import DownloaderStrategy
from agrigee_lite.ee_utils import (
    ee_gdf_to_feature_collection,
    ee_get_tasks_status,
    get_number_of_available_service_accounts,
    login_with_service_account_n,
)
from agrigee_lite.misc import (
    create_dict_hash,
    create_gdf_hash,
    get_reducer_names,
    log_dict_function_call_summary,
    quadtree_clustering,
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
    cluster_size: int,
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
    cluster_size : int
        Maximum size for spatial clustering.
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

    gdf = quadtree_clustering(gdf, max_size=cluster_size)

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


def download_multiple_sits(  # noqa: C901
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None = None,
    original_index_column_name: str = "original_index",
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
    subsampling_max_pixels: float = 1_000,
    chunksize: int = 10,
    max_parallel_downloads: int = 38,
    max_retries_per_chunk: int = 1,
    force_redownload: bool = False,
) -> pd.DataFrame:
    if len(gdf) == 0:
        return pd.DataFrame()

    gdf = sanitize_and_prepare_input_gdf(
        gdf, satellite, original_index_column_name, 1000, start_date_column_name, end_date_column_name
    )

    # --- SpatiaLite cache: check per feature, keep only uncached rows ---
    _engine = get_engine()
    cached_frames: list[pd.DataFrame] = []
    if _engine is not None and not force_redownload and len(gdf) > 0:
        uncached_positions: list[Any] = []
        for pos, row in gdf.iterrows():
            s = str(row[start_date_column_name])[:10]
            e = str(row[end_date_column_name])[:10]
            hit = fetch_sits(_engine, row.geometry, s, e, satellite, reducers, subsampling_max_pixels)
            if hit is not None:
                hit[original_index_column_name] = row[original_index_column_name]
                cached_frames.append(hit)
            else:
                uncached_positions.append(pos)
        gdf = gdf.loc[uncached_positions].reset_index(drop=True)
        if gdf.empty:
            logging.debug("Cache: all %d features already cached.", len(cached_frames))
            return pd.concat(cached_frames, ignore_index=True) if cached_frames else pd.DataFrame()
    # --- end cache check ---

    metadata_dict: dict[str, Any] = {}
    metadata_dict |= log_dict_function_call_summary(["gdf", "satellite", "max_parallel_downloads", "force_redownload"])
    metadata_dict |= satellite.log_dict()

    output_path = (
        pathlib.Path.home()
        / ".cache"
        / "agrigee_lite"
        / "sits"
        / f"{create_gdf_hash(gdf, start_date_column_name, end_date_column_name)}_{create_dict_hash(metadata_dict)}"
    )

    if force_redownload:
        for f in output_path.glob("*"):
            f.unlink()

    output_path.mkdir(parents=True, exist_ok=True)

    downloader = DownloaderStrategy(download_folder=output_path)

    num_chunks = (len(gdf) + chunksize - 1) // chunksize

    already_downloaded_files = [int(x.stem) for x in output_path.glob("*.csv")]
    print(output_path, "-", len(already_downloaded_files), "chunks already downloaded and will be skipped.")
    initial_download_chunks = sorted(set(range(num_chunks)) - set(already_downloaded_files))

    pbar = tqdm(
        total=len(initial_download_chunks) * chunksize,
        unit="feature",
        smoothing=0,
        bar_format="{percentage:3.0f}% | {n_fmt}/{total_fmt} | [{elapsed}<{remaining}, {rate_fmt}, {postfix}]",
    )

    def fetch_chunk_url(chunk_id: int) -> tuple[int, str]:
        import socket
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(90)
        try:
            sub = gdf.iloc[chunk_id * chunksize : (chunk_id + 1) * chunksize]
            ee_expression = build_ee_expression(
                sub,
                satellite,
                reducers,
                subsampling_max_pixels,
                original_index_column_name,
                start_date_column_name,
                end_date_column_name,
            )
            url = ee_expression.getDownloadURL(
                filetype="csv",
                selectors=build_selectors(satellite, reducers),
                filename=f"{chunk_id}",
            )
            return chunk_id, url
        finally:
            socket.setdefaulttimeout(old_timeout)

    to_download_chunks = list(initial_download_chunks)
    not_sent_to_server: list[int | str] = []

    num_accounts = get_number_of_available_service_accounts()
    current_account_idx = 0
    account_active_chunks: dict[int, set[int | str]] = defaultdict(set)

    def update_pbar(snap: dict[int | str, Any]) -> None:
        stats = downloader.stats_from_snapshot(snap)
        pbar.n = stats["completed"] * chunksize
        pbar.refresh()
        active_by_account = {acc: len(chunks) for acc, chunks in account_active_chunks.items() if chunks}
        pbar.set_postfix({
            "gee_er": len(not_sent_to_server),
            "ar2_er": stats["errors"],
            **{f"{acc}": count for acc, count in active_by_account.items()},
        })

    download_with_too_many_errors = 0
    # ThreadPoolExecutor lives for the entire download session — no per-batch
    # creation/destruction overhead.
    with ThreadPoolExecutor(max_workers=7) as executor:
        while (True):
            # Single RPC call covers all status checks this iteration.
            snap = downloader._get_downloads_snapshot()
            stats = downloader.stats_from_snapshot(snap)

            if (stats["completed"] + download_with_too_many_errors) == len(initial_download_chunks):
                break

            failed_within_limit: list[int | str] = [
                my_id
                for my_id, d in snap.items()
                if d.status == "error" and downloader.retry_count.get(my_id, 0) < max_retries_per_chunk
            ]
            for chunk_id in failed_within_limit:
                downloader.increment_retry_count(chunk_id)

            if failed_within_limit or not_sent_to_server:
                to_download_chunks = sorted(
                    set(to_download_chunks) | set(failed_within_limit) | set(not_sent_to_server)
                )
                not_sent_to_server = []

            # Only switch accounts and dispatch when there are chunks waiting.
            # While aria2 is draining the queue, skip login and account rotation
            # entirely to avoid pointless credential churn.
            if to_download_chunks:
                login_with_service_account_n(current_account_idx)

                # Retire finished/failed chunks from this account's active set.
                current_active_set = account_active_chunks[current_account_idx]
                current_active_set -= {
                    cid for cid in current_active_set
                    if (output_path / f"{cid}.csv").exists()
                    or cid in failed_within_limit
                    or cid in not_sent_to_server
                }

                available_slots = max_parallel_downloads - len(current_active_set)

                if available_slots > 0:
                    batch_size = min(available_slots, len(to_download_chunks))
                    current_batch = to_download_chunks[:batch_size]
                    to_download_chunks = to_download_chunks[batch_size:]

                    new_downloads: list[tuple[int | str, str]] = []
                    future_to_chunk = {executor.submit(fetch_chunk_url, int(cid)): cid for cid in current_batch}

                    try:
                        for future in as_completed(future_to_chunk, timeout=120):
                            cid = future_to_chunk[future]
                            try:
                                new_downloads.append(future.result())
                            except KeyboardInterrupt:
                                pbar.close()
                                raise
                            except Exception:
                                logging.exception(output_path, "- Chunk id =", cid, " - Failed to get download URL.")
                                not_sent_to_server.append(cid)
                    except TimeoutError:
                        for f, cid in future_to_chunk.items():
                            if not f.done():
                                logging.warning("%s - Chunk id=%s - GEE getDownloadURL timed out, requeuing.", output_path, cid)
                                not_sent_to_server.append(cid)

                    if new_downloads:
                        downloader.add_download(new_downloads)
                        account_active_chunks[current_account_idx].update(d[0] for d in new_downloads)

                current_account_idx = (current_account_idx + 1) % num_accounts
                time.sleep(0.5)
            else:
                # All chunks are in-flight with aria2 — just poll at a slower
                # cadence without touching accounts.
                time.sleep(2.0)

            download_with_too_many_errors = sum(
                1 for my_id, d in snap.items()
                if d.status == "error" and downloader.retry_count.get(my_id, 0) >= max_retries_per_chunk
            )
            update_pbar(snap)

    pbar.close()

    csv_files = sorted(output_path.glob("*.csv"), key=lambda x: int(x.stem))
    frames = [pd.read_csv(f) for f in tqdm(csv_files, desc="Combining downloaded chunks") if f.stat().st_size > 0]
    whole_result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    whole_result_df = prepare_output_df(whole_result_df, satellite, original_index_column_name)

    # --- SpatiaLite cache: persist newly downloaded data ---
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
    # --- end cache write ---

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
    cluster_size: int = 500,
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
    cluster_size : int, optional
        Maximum cluster size for spatial grouping, by default 500.
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
        gdf, satellite, original_index_column_name, cluster_size, start_date_column_name, end_date_column_name
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
        sorted(gdf.cluster_id.unique()), desc=f"Creating GEE tasks ({satellite.shortName}_{hashname}_{cluster_size})"
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
    cluster_size: int = 500,
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
    cluster_size : int, optional
        Maximum cluster size for spatial grouping, by default 500.
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
        gdf, satellite, original_index_column_name, cluster_size, start_date_column_name, end_date_column_name
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


def _fetch_sits_chunk_url_sync(
    chunk_id: int,
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    original_index_column_name: str,
    start_date_column_name: str,
    end_date_column_name: str,
    chunksize: int,
) -> tuple[int, str]:
    """Resolve the download URL for a single GEE SITS chunk (blocking, runs in a thread)."""
    import socket
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(90)
    try:
        sub = gdf.iloc[chunk_id * chunksize : (chunk_id + 1) * chunksize]
        ee_expression = build_ee_expression(
            sub,
            satellite,
            reducers,
            subsampling_max_pixels,
            original_index_column_name,
            start_date_column_name,
            end_date_column_name,
        )
        url = ee_expression.getDownloadURL(
            filetype="csv",
            selectors=build_selectors(satellite, reducers),
            filename=f"{chunk_id}",
        )
        return chunk_id, url
    finally:
        socket.setdefaulttimeout(old_timeout)


async def download_multiple_sits_async(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None = None,
    original_index_column_name: str = "original_index",
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
    subsampling_max_pixels: float = 1_000,
    chunksize: int = 10,
    max_parallel_downloads: int = 38,
    max_retries_per_chunk: int = 1,
    force_redownload: bool = False,
) -> pd.DataFrame:
    """
    Async version of :func:`download_multiple_sits`.

    GEE URL fetches are issued concurrently via ``asyncio.to_thread``; aria2
    progress is polled with ``asyncio.sleep`` so the event loop stays free.
    Multiple coroutines can run in parallel without blocking each other.
    """
    if len(gdf) == 0:
        return pd.DataFrame()

    gdf = sanitize_and_prepare_input_gdf(
        gdf, satellite, original_index_column_name, 1000, start_date_column_name, end_date_column_name
    )

    metadata_dict: dict[str, Any] = {}
    metadata_dict |= log_dict_function_call_summary(["gdf", "satellite", "max_parallel_downloads", "force_redownload"])
    metadata_dict |= satellite.log_dict()

    output_path = (
        pathlib.Path.home()
        / ".cache"
        / "agrigee_lite"
        / "sits"
        / f"{create_gdf_hash(gdf, start_date_column_name, end_date_column_name)}_{create_dict_hash(metadata_dict)}"
    )

    if force_redownload:
        for f in output_path.glob("*"):
            f.unlink()

    output_path.mkdir(parents=True, exist_ok=True)

    downloader = DownloaderStrategy(download_folder=output_path)
    num_chunks = (len(gdf) + chunksize - 1) // chunksize
    already_downloaded = [int(x.stem) for x in output_path.glob("*.csv")]
    print(output_path, "-", len(already_downloaded), "chunks already downloaded and will be skipped.")
    pending_chunks = sorted(set(range(num_chunks)) - set(already_downloaded))

    pbar = tqdm(
        total=len(pending_chunks) * chunksize,
        unit="feature",
        smoothing=0,
        bar_format="{percentage:3.0f}% | {n_fmt}/{total_fmt} | [{elapsed}<{remaining}, {rate_fmt}, {postfix}]",
    )

    def update_pbar() -> None:
        pbar.n = downloader.num_completed_downloads * chunksize
        pbar.refresh()

    not_sent_to_server: list[Any] = []
    download_with_too_many_errors = 0
    initial_pending_count = len(pending_chunks)
    to_fetch = list(pending_chunks)

    while (downloader.num_completed_downloads + download_with_too_many_errors) != initial_pending_count:
        failed_within_limit = downloader.get_failed_downloads_within_retry_limit(max_retries_per_chunk)
        for cid in failed_within_limit:
            downloader.increment_retry_count(cid)

        if failed_within_limit or not_sent_to_server:
            to_fetch = sorted(set(to_fetch) | set(failed_within_limit) | set(not_sent_to_server))
            not_sent_to_server = []

        available = [
            cid for cid in to_fetch if not (output_path / f"{cid}.csv").exists() and cid not in failed_within_limit
        ]
        batch = available[: max_parallel_downloads - downloader.num_unfinished_downloads]
        to_fetch = [cid for cid in to_fetch if cid not in batch]

        if batch:
            fetch_results = await asyncio.gather(
                *[
                    asyncio.to_thread(
                        _fetch_sits_chunk_url_sync,
                        cid,
                        gdf,
                        satellite,
                        reducers,
                        subsampling_max_pixels,
                        original_index_column_name,
                        start_date_column_name,
                        end_date_column_name,
                        chunksize,
                    )
                    for cid in batch
                ],
                return_exceptions=True,
            )

            new_downloads: list[tuple[int | str, str]] = []
            for cid, res in zip(batch, fetch_results, strict=False):
                if isinstance(res, BaseException):
                    logging.exception(f"{output_path} - Chunk {cid} - Failed to get URL.")
                    not_sent_to_server.append(cid)
                else:
                    new_downloads.append(res)

            if new_downloads:
                downloader.add_download(new_downloads)

        update_pbar()
        download_with_too_many_errors = downloader.get_num_failed_downloads_exceeding_retry_limit(max_retries_per_chunk)
        await asyncio.sleep(0.5)

    pbar.close()

    csv_files = sorted(output_path.glob("*.csv"), key=lambda x: int(x.stem))
    frames = [pd.read_csv(f) for f in tqdm(csv_files, desc="Combining downloaded chunks") if f.stat().st_size > 0]
    whole_result_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return prepare_output_df(whole_result_df, satellite, original_index_column_name)
