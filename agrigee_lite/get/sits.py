import concurrent.futures
import getpass
import json
import logging
import logging.handlers
import pathlib
import queue
from functools import partial

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import pandera as pa
from shapely import Polygon
from smart_open import open  # noqa: A004
from tqdm.std import tqdm

from agrigee_lite.ee_utils import ee_gdf_to_feature_collection, ee_get_tasks_status
from agrigee_lite.misc import (
    add_indexnum_column,
    create_gdf_hash,
    log_dict_function_call_summary,
    quadtree_clustering,
    remove_underscore_in_df,
)
from agrigee_lite.sat.abstract_satellite import AbstractSatellite
from agrigee_lite.task_manager import GEETaskManager


# @cached # Doesn't work with lists as parameters :(
def download_single_sits(
    geometry: Polygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
) -> pd.DataFrame:
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

    if len(sits_df) == 0:
        return pd.DataFrame()
    else:
        sits_df = sits_df.drop(columns=["geo"])
        remove_underscore_in_df(sits_df)

        if "timestamp" in sits_df.columns:
            sits_df["timestamp"] = pd.to_datetime(sits_df["timestamp"])

        sits_df.fillna(0, inplace=True)

        band_columns = sorted(set(sits_df.columns.tolist()) - {"timestamp", "validPixelsCount"})
        sits_df = sits_df[~(sits_df[band_columns] == 0).all(axis=1)]

        return sits_df


def download_multiple_sits(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
) -> pd.DataFrame:
    add_indexnum_column(gdf)

    fc = ee_gdf_to_feature_collection(gdf)
    ee_expression = ee.FeatureCollection(
        fc.map(
            partial(
                satellite.compute,
                reducers=reducers,
                subsampling_max_pixels=subsampling_max_pixels,
            )
        )
    ).flatten()
    sits_df = ee.data.computeFeatures({"expression": ee_expression, "fileFormat": "PANDAS_DATAFRAME"})

    if len(sits_df) == 0:
        return pd.DataFrame()
    else:
        sits_df = sits_df.drop(columns=["geo"])
        remove_underscore_in_df(sits_df)

        return sits_df


def download_multiple_sits_multithread(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
    mini_chunksize: int = 10,
    num_threads_rush: int = 30,
    num_threads_retry: int = 10,
    pbar: tqdm | None = None,
) -> pd.DataFrame:
    add_indexnum_column(gdf)
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

    indexes_with_errors: list[int] = []
    whole_result_df = pd.DataFrame()
    num_chunks = (len(gdf) + mini_chunksize - 1) // mini_chunksize
    all_chunk_ids = list(range(num_chunks))

    def process_download(gdf_chunk: gpd.GeoDataFrame, i: int) -> tuple[pd.DataFrame, int]:
        try:
            result_chunk = download_multiple_sits(
                gdf_chunk,
                satellite,
                reducers=reducers,
                subsampling_max_pixels=subsampling_max_pixels,
            )
            return result_chunk, i  # noqa: TRY300
        except KeyboardInterrupt:
            raise KeyboardInterrupt  # noqa: B904
        except Exception as e:
            logger.error(f"download_multiple_sits_multithread_{i}_{satellite.shortName} = {e}")  # noqa: TRY400
            return pd.DataFrame(), i

    def run_downloads(chunk_ids: list[int], num_threads: int) -> None:
        nonlocal whole_result_df
        error_count = 0

        with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
            futures = [
                executor.submit(
                    partial(
                        process_download,
                        gdf.iloc[i * mini_chunksize : (i + 1) * mini_chunksize].reset_index(drop=True),
                        i,
                    )
                )
                for i in chunk_ids
            ]

            for future in concurrent.futures.as_completed(futures):
                result_df_chunk, i = future.result()
                if result_df_chunk.empty:
                    error_count += 1
                    indexes_with_errors.append(i)

                    if pbar is not None:
                        pbar.set_postfix({"errors": error_count})
                else:
                    whole_result_df = pd.concat([whole_result_df, result_df_chunk])

                    if pbar is not None:
                        pbar.update(result_df_chunk.indexnum.nunique())

    run_downloads(all_chunk_ids, num_threads=num_threads_rush)

    if indexes_with_errors:
        run_downloads(indexes_with_errors, num_threads=num_threads_retry)

    return whole_result_df


# Dead code, but kept in case we need it in the future
async def __download_multiple_sits_async(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
    mini_chunksize: int = 10,
    initial_concurrency: int = 30,
    retry_concurrency: int = 10,
    initial_timeout: float = 20,
    retry_timeout: float = 10,
) -> pd.DataFrame:
    import anyio

    add_indexnum_column(gdf)
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

    num_chunks = (len(gdf) + mini_chunksize - 1) // mini_chunksize
    all_chunk_ids = list(range(num_chunks))
    whole_result_df = pd.DataFrame()
    indexes_with_errors: list[int] = []

    async def run_download(chunk_ids: list[int], concurrency: int, timeout: float) -> None:
        sem = anyio.Semaphore(concurrency)

        async def download_task(chunk_id: int) -> None:
            nonlocal whole_result_df

            async with sem:
                start_idx = chunk_id * mini_chunksize
                end_idx = min(start_idx + mini_chunksize, len(gdf))

                gdf_chunk = gdf.iloc[start_idx:end_idx].reset_index(drop=True)

                try:
                    with anyio.fail_after(timeout):
                        chunk_result_df = await anyio.to_thread.run_sync(
                            partial(
                                download_multiple_sits,
                                gdf_chunk,
                                satellite,
                                reducers=reducers,
                                subsampling_max_pixels=subsampling_max_pixels,
                            )
                        )

                    whole_result_df = pd.concat([whole_result_df, chunk_result_df])
                except KeyboardInterrupt:
                    raise KeyboardInterrupt  # noqa: B904
                except Exception as e:
                    logger.error(f"download_multiple_sits_anyio_{chunk_id}_{satellite.shortName} = {e}")  # noqa: TRY400
                    indexes_with_errors.append(chunk_id)

        async with anyio.create_task_group() as tg:
            for cid in chunk_ids:
                tg.start_soon(download_task, cid)

    await run_download(all_chunk_ids, concurrency=initial_concurrency, timeout=initial_timeout)

    if indexes_with_errors:
        await run_download(indexes_with_errors, concurrency=retry_concurrency, timeout=retry_timeout)

    queue_listener.stop()

    whole_result_df = whole_result_df.sort_values(by=["indexnum"], kind="stable").reset_index(drop=True)

    return whole_result_df


def download_multiple_sits_chunks_multithread(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
    chunksize: int = 10000,
    mini_chunksize: int = 10,
    initial_concurrency: int = 30,
    retry_concurrency: int = 10,
    force_redownload: bool = False,
) -> gpd.GeoDataFrame:
    if len(gdf) == 0:
        print("Empty GeoDataFrame, nothing to download")
        return pd.DataFrame()

    schema = pa.DataFrameSchema({
        "geometry": pa.Column("geometry", nullable=False),
        "start_date": pa.Column(
            pa.DateTime,
            nullable=False,
            # checks=pa.Check.in_range(min_value=satellite.startDate, max_value=satellite.endDate),
        ),
        "end_date": pa.Column(
            pa.DateTime,
            nullable=False,
            # checks=pa.Check.in_range(min_value=satellite.startDate, max_value=satellite.endDate),
        ),
    })
    schema.validate(gdf, lazy=True)

    gdf = gdf.copy()
    add_indexnum_column(gdf)

    hashlib_gdf = create_gdf_hash(gdf)
    output_path = pathlib.Path("data/temp") / f"{satellite.shortName}_{hashlib_gdf}_{chunksize}"
    output_path.mkdir(parents=True, exist_ok=True)
    existing_chunks = {int(f.stem) for f in output_path.glob("*.parquet") if f.stem.isdigit()}

    gdf = quadtree_clustering(gdf, max_size=1000)

    num_chunks = (len(gdf) + chunksize - 1) // chunksize

    with tqdm(total=len(gdf), desc="Downloading multiple sits", smoothing=0.5) as pbar:
        for current_chunk in range(num_chunks):
            if (not force_redownload) and (current_chunk in existing_chunks):
                continue

            chunk_df = download_multiple_sits_multithread(
                gdf.iloc[current_chunk * chunksize : (current_chunk + 1) * chunksize],
                satellite,
                reducers=reducers,
                subsampling_max_pixels=subsampling_max_pixels,
                mini_chunksize=mini_chunksize,
                num_threads_rush=initial_concurrency,
                num_threads_retry=retry_concurrency,
                pbar=pbar,
            )

            if "timestamp" in chunk_df.columns:
                chunk_df["timestamp"] = pd.to_datetime(chunk_df.timestamp)

            for col in chunk_df.columns:
                if col not in ["timestamp", "indexnum"]:
                    chunk_df[col] = chunk_df[col].astype(np.float16)

            chunk_df.to_parquet(f"{output_path / str(current_chunk)}.parquet")

    whole_result_df = pd.DataFrame()
    for f in sorted(output_path.glob("*.parquet"), key=lambda p: int(p.stem)):
        df = pd.read_parquet(f)
        whole_result_df = pd.concat([whole_result_df, df], ignore_index=True)

    if len(whole_result_df) != 0:
        whole_result_df = whole_result_df.sort_values(by=["indexnum"], kind="stable").reset_index(drop=True)

    return whole_result_df


def download_multiple_sits_task_gdrive(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    file_stem: str,
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
    taskname: str = "",
    gee_save_folder: str = "GEE_EXPORTS",
) -> ee.batch.Task:
    if taskname == "":
        taskname = file_stem

    add_indexnum_column(gdf)
    fc = ee_gdf_to_feature_collection(gdf)
    ee_expression = ee.FeatureCollection(
        fc.map(
            partial(
                satellite.compute,
                reducers=reducers,
                subsampling_max_pixels=subsampling_max_pixels,
            )
        )
    ).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=ee_expression,
        description=taskname,
        fileFormat="CSV",
        fileNamePrefix=file_stem,
        folder=gee_save_folder,
        selectors=[
            "00_indexnum",
            "01_timestamp",
            *[numeral_band_name for _, numeral_band_name in satellite.selectedBands],
            *[numeral_indice_name for _, _, numeral_indice_name in satellite.selectedIndices],
            "99_validPixelsCount",
        ],
    )

    return task


def download_multiple_sits_task_gcs(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    bucket_name: str,
    file_path: str,
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
    taskname: str = "",
) -> ee.batch.Task:
    if taskname == "":
        taskname = file_path

    add_indexnum_column(gdf)
    fc = ee_gdf_to_feature_collection(gdf)
    ee_expression = ee.FeatureCollection(
        fc.map(
            partial(
                satellite.compute,
                reducers=reducers,
                subsampling_max_pixels=subsampling_max_pixels,
            )
        )
    ).flatten()

    task = ee.batch.Export.table.toCloudStorage(
        bucket=bucket_name,
        collection=ee_expression,
        description=taskname,
        fileFormat="CSV",
        fileNamePrefix=file_path,
        selectors=[
            "00_indexnum",
            "01_timestamp",
            *[numeral_band_name for _, numeral_band_name in satellite.selectedBands],
            *[numeral_indice_name for _, _, numeral_indice_name in satellite.selectedIndices],
            "99_validPixelsCount",
        ],
    )

    return task


def download_multiple_sits_chunks_gdrive(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
    cluster_size: int = 500,
    gee_save_folder: str = "GEE_EXPORTS",
    force_redownload: bool = False,
    wait: bool = True,
) -> None:
    if len(gdf) == 0:
        print("Empty GeoDataFrame, nothing to download")
        return None

    schema = pa.DataFrameSchema({
        "geometry": pa.Column("geometry", nullable=False),
        "start_date": pa.Column(
            pa.DateTime,
            nullable=False,
            checks=pa.Check.in_range(min_value=satellite.startDate, max_value=satellite.endDate),
        ),
        "end_date": pa.Column(
            pa.DateTime,
            nullable=False,
            checks=pa.Check.in_range(min_value=satellite.startDate, max_value=satellite.endDate),
        ),
    })
    schema.validate(gdf, lazy=True)

    task_mgr = GEETaskManager()  # To handle the new tasks

    tasks_df = ee_get_tasks_status()
    tasks_df = tasks_df[tasks_df.description.str.startswith("agl_multiple_sits_")]
    completed_or_running_tasks = set(
        tasks_df.description.apply(lambda x: x.split("_", 1)[0] + "_" + x.split("_", 2)[2]).tolist()
    )  # The task is the same, no matter who started it

    add_indexnum_column(gdf)
    gdf = quadtree_clustering(gdf, cluster_size)
    username = getpass.getuser().replace("_", "")
    hashname = create_gdf_hash(gdf)

    for cluster_id in tqdm(sorted(gdf.cluster_id.unique())):
        cluster_id = int(cluster_id)

        if (force_redownload) or (
            f"agl_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}" not in completed_or_running_tasks
        ):
            task = download_multiple_sits_task_gdrive(
                gdf[gdf.cluster_id == cluster_id],
                satellite,
                f"{satellite.shortName}_{hashname}_{cluster_id}",
                reducers=reducers,
                subsampling_max_pixels=subsampling_max_pixels,
                taskname=f"agl_{username}_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}",
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
    reducers: list[str] | None = None,
    subsampling_max_pixels: float = 1_000,
    cluster_size: int = 500,
    force_redownload: bool = False,
    wait: bool = True,
) -> None | pd.DataFrame:
    if len(gdf) == 0:
        print("Empty GeoDataFrame, nothing to download")
        return None

    schema = pa.DataFrameSchema({
        "geometry": pa.Column("geometry", nullable=False),
        "start_date": pa.Column(
            pa.DateTime,
            nullable=False,
            checks=pa.Check.in_range(min_value=satellite.startDate, max_value=satellite.endDate),
        ),
        "end_date": pa.Column(
            pa.DateTime,
            nullable=False,
            checks=pa.Check.in_range(min_value=satellite.startDate, max_value=satellite.endDate),
        ),
    })
    schema.validate(gdf, lazy=True)

    task_mgr = GEETaskManager()
    tasks_df = ee_get_tasks_status()
    tasks_df = tasks_df[tasks_df.description.str.startswith("agl_")].reset_index(drop=True)
    completed_or_running_tasks = set(
        tasks_df.description.apply(lambda x: x.split("_", 1)[0] + "_" + x.split("_", 2)[2]).tolist()
    )  # The task is the same, no matter who started it

    add_indexnum_column(gdf)
    gdf = quadtree_clustering(gdf, cluster_size)
    username = getpass.getuser().replace("_", "")
    hashname = create_gdf_hash(gdf)

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
                reducers=reducers,
                subsampling_max_pixels=subsampling_max_pixels,
                bucket_name=bucket_name,
                file_path=f"{gcs_save_folder}/{cluster_id}",
                taskname=f"agl_{username}_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}",
            )

            task_mgr.add(task)

        file_uris.append(f"gs://{bucket_name}/{gcs_save_folder}/{cluster_id}.csv")

    task_mgr.start()

    if wait:
        task_mgr.wait()

        df = pd.DataFrame()
        for file_uri in file_uris:
            with open(file_uri, "rb") as f:
                sub_df = pd.read_csv(f)
                df = pd.concat([df, sub_df], ignore_index=True)

        remove_underscore_in_df(df)
        return df
    else:
        return None
