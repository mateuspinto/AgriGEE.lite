import getpass
import json
import logging
import pathlib
import time
from functools import partial

import ee
import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
from shapely import MultiPolygon, Point, Polygon
from tqdm.std import tqdm

from agrigee_lite.downloader import DownloaderStrategy
from agrigee_lite.ee_utils import ee_gdf_to_feature_collection, ee_get_tasks_status
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

    ee_feature = ee.Feature(
        geometry.__geo_interface__,
        {"s": start_date, "e": end_date, "0": 0},
    )
    ee_expression = satellite.compute(ee_feature, reducers=reducers, subsampling_max_pixels=subsampling_max_pixels)

    sits_df = ee.data.computeFeatures({"expression": ee_expression, "fileFormat": "PANDAS_DATAFRAME"})
    sits_df = prepare_output_df(sits_df, satellite, "IGNORED")

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
    max_parallel_downloads: int = 40,
    max_retries_per_chunk: int = 1,
    force_redownload: bool = False,
) -> pd.DataFrame:
    """
    Download satellite time series for multiple geometries using parallel processing.

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
    chunksize : int, optional
        Number of features to process per chunk, by default 10.
    max_parallel_downloads : int, optional
        Maximum number of parallel downloads, by default 40.
    max_retries_per_chunk : int, optional
        Maximum number of retry attempts per chunk, by default 1.
    force_redownload : bool, optional
        Whether to force re-download of existing data, by default False.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame containing satellite time series for all geometries.
    """
    if len(gdf) == 0:
        return pd.DataFrame()

    gdf = sanitize_and_prepare_input_gdf(
        gdf, satellite, original_index_column_name, 1000, start_date_column_name, end_date_column_name
    )

    metadata_dict: dict[str, str] = {}
    metadata_dict |= log_dict_function_call_summary(["gdf", "satellite", "max_parallel_downloads", "force_redownload"])
    metadata_dict |= satellite.log_dict()

    output_path = (
        pathlib.Path("data/temp/sits")
        / f"{create_gdf_hash(gdf, start_date_column_name, end_date_column_name)}_{create_dict_hash(metadata_dict)}"
    )

    if force_redownload:
        for f in output_path.glob("*"):
            f.unlink()

    output_path.mkdir(parents=True, exist_ok=True)

    downloader = DownloaderStrategy(download_folder=output_path)

    num_chunks = (len(gdf) + chunksize - 1) // chunksize

    already_downloaded_files = [int(x.stem) for x in output_path.glob("*.csv")]
    logging.info(output_path, "-", len(already_downloaded_files), "chunks already downloaded and will be skipped.")
    initial_download_chunks = sorted(set(range(num_chunks)) - set(already_downloaded_files))

    pbar = tqdm(
        total=len(initial_download_chunks) * chunksize,
        desc=f"Building download URLs ({output_path.name})",
        unit="feature",
        smoothing=0,
    )

    def update_pbar():
        pbar.n = downloader.num_completed_downloads * chunksize
        pbar.refresh()
        pbar.set_postfix({
            "not_sent_to_server": len(not_sent_to_server),
            "aria2_errors": downloader.num_downloads_with_error,
            "active_downloads": downloader.num_unfinished_downloads,
        })

    to_download_chunks = initial_download_chunks
    while downloader.num_completed_downloads != len(initial_download_chunks):
        not_sent_to_server = []

        for current_chunk in to_download_chunks:
            while downloader.num_unfinished_downloads >= max_parallel_downloads:
                time.sleep(1)
                update_pbar()

            sub = gdf.iloc[current_chunk * chunksize : (current_chunk + 1) * chunksize]
            ee_expression = build_ee_expression(
                sub,
                satellite,
                reducers,
                subsampling_max_pixels,
                original_index_column_name,
                start_date_column_name,
                end_date_column_name,
            )

            try:
                url = ee_expression.getDownloadURL(
                    filetype="csv",
                    selectors=build_selectors(satellite, reducers),
                    filename=f"{current_chunk}",
                )

                downloader.add_download([(current_chunk, url)])

            except KeyboardInterrupt:
                pbar.close()
                raise

            except ee.ee_exception.EEException:
                logging.exception(output_path, "- Chunk id =", current_chunk, " - Failed to get download URL.")
                not_sent_to_server.append(current_chunk)

            update_pbar()

        failed_within_limit = downloader.get_failed_downloads_within_retry_limit(max_retries_per_chunk)
        for chunk_id in failed_within_limit:
            downloader.increment_retry_count(chunk_id)

        to_download_chunks = sorted(set(not_sent_to_server) | set(failed_within_limit))
        while len(to_download_chunks) == 0 and (downloader.num_completed_downloads != len(initial_download_chunks)):
            time.sleep(1)
            failed_within_limit = downloader.get_failed_downloads_within_retry_limit(max_retries_per_chunk)
            for chunk_id in failed_within_limit:
                downloader.increment_retry_count(chunk_id)
            to_download_chunks = sorted(set(not_sent_to_server) | set(failed_within_limit))
            update_pbar()

    pbar.close()
    whole_result_df = pd.DataFrame()
    for f in output_path.glob("*.csv"):
        df = pd.read_csv(f)
        whole_result_df = pd.concat([whole_result_df, df], ignore_index=True)

    whole_result_df = prepare_output_df(whole_result_df, satellite, original_index_column_name)

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
    from smart_open import open  # noqa: A004

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
    metadata_dict: dict[str, str] = {}
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
