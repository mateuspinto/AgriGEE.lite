import concurrent.futures
import getpass
import logging
import logging.handlers
import pathlib
import queue
from functools import partial

import anyio
import ee
import geopandas as gpd
import pandas as pd
import pandera as pa
from shapely import Polygon
from tqdm.std import tqdm

from agrigee_lite.ee_utils import ee_gdf_to_feature_collection, ee_get_tasks_status
from agrigee_lite.misc import (
    cached,
    create_gdf_hash,
    long_to_wide_dataframe,
    quadtree_clustering,
    remove_underscore_in_df,
)
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


@cached
def single_sits(
    geometry: Polygon, start_date: pd.Timestamp | str, end_date: pd.Timestamp | str, satellite: AbstractSatellite
) -> pd.DataFrame:
    start_date = start_date.strftime("%Y-%m-%d") if isinstance(start_date, pd.Timestamp) else start_date
    end_date = end_date.strftime("%Y-%m-%d") if isinstance(end_date, pd.Timestamp) else end_date

    ee_feature = ee.Feature(
        ee.Geometry(geometry.__geo_interface__),
        {"start_date": start_date, "end_date": end_date, "00_indexnum": 0},
    )
    ee_expression = satellite.compute(ee_feature)

    sits_df = ee.data.computeFeatures({"expression": ee_expression, "fileFormat": "PANDAS_DATAFRAME"}).drop(
        columns=["geo"]
    )

    remove_underscore_in_df(sits_df)
    sits_df = long_to_wide_dataframe(sits_df, satellite.shortName)

    return sits_df


def multiple_sits(gdf: gpd.GeoDataFrame, satellite: AbstractSatellite) -> pd.DataFrame:
    fc = ee_gdf_to_feature_collection(gdf)
    ee_expression = ee.FeatureCollection(fc.map(satellite.compute)).flatten()
    sits_df = ee.data.computeFeatures({"expression": ee_expression, "fileFormat": "PANDAS_DATAFRAME"}).drop(
        columns=["geo"]
    )

    remove_underscore_in_df(sits_df)
    sits_df = long_to_wide_dataframe(sits_df, satellite.shortName)

    return sits_df


def multiple_sits_multithread(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    mini_chunksize: int = 10,
    num_threads_rush: int = 30,
    num_threads_retry: int = 10,
) -> pd.DataFrame:
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
            result_chunk = multiple_sits(gdf_chunk, satellite)
            result_chunk["chunk_id"] = i
            return result_chunk, i  # noqa: TRY300
        except Exception as e:
            logger.error(f"download_multiple_sits_multithread_{i}_{satellite.shortName} = {e}")  # noqa: TRY400
            return pd.DataFrame(), i

    def run_downloads(chunk_ids: list[int], num_threads: int, desc: str) -> None:
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

            pbar = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=desc)
            for future in pbar:
                result_df_chunk, i = future.result()
                if result_df_chunk.empty:
                    error_count += 1
                    indexes_with_errors.append(i)
                    pbar.set_postfix({"errors": error_count})
                else:
                    whole_result_df = pd.concat([whole_result_df, result_df_chunk])

    run_downloads(all_chunk_ids, num_threads=num_threads_rush, desc="Downloading")

    if indexes_with_errors:
        run_downloads(
            indexes_with_errors,
            num_threads=num_threads_retry,
            desc="Re-running failed downloads",
        )

    whole_result_df["00_indexnum"] = (
        whole_result_df["chunk_id"] * (whole_result_df["00_indexnum"].max() + 1) + whole_result_df["00_indexnum"]
    )
    whole_result_df.drop(columns=["chunk_id"], inplace=True)
    whole_result_df = whole_result_df.sort_values("00_indexnum", kind="stable").reset_index(drop=True)

    remove_underscore_in_df(whole_result_df)
    whole_result_df = long_to_wide_dataframe(whole_result_df)

    return whole_result_df


async def multiple_sits_async(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    mini_chunksize: int = 10,
    initial_concurrency: int = 30,
    retry_concurrency: int = 10,
    initial_timeout: float = 20,
    retry_timeout: float = 10,
) -> pd.DataFrame:
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
                        chunk_result_df = await anyio.to_thread.run_sync(partial(multiple_sits, gdf_chunk, satellite))
                        chunk_result_df["chunk_id"] = chunk_id

                    whole_result_df = pd.concat([whole_result_df, chunk_result_df])

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

    whole_result_df["00_indexnum"] = (
        whole_result_df["chunk_id"] * (whole_result_df["00_indexnum"].max() + 1) + whole_result_df["00_indexnum"]
    )
    whole_result_df.drop(columns=["chunk_id"], inplace=True)
    whole_result_df = whole_result_df.sort_values("00_indexnum", kind="stable").reset_index(drop=True)

    remove_underscore_in_df(whole_result_df)
    whole_result_df = long_to_wide_dataframe(whole_result_df)

    return whole_result_df


def multiple_sits_chunks_multithread(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    chunksize: int = 10000,
    mini_chunksize: int = 10,
    initial_concurrency: int = 30,
    retry_concurrency: int = 10,
) -> gpd.GeoDataFrame:
    if len(gdf) == 0:
        print("Empty GeoDataFrame, nothing to download")
        return pd.DataFrame()

    schema = pa.DataFrameSchema({
        "geometry": pa.Column("geometry", nullable=False),
        "start_date": pa.Column(pa.DateTime, nullable=False),
        "end_date": pa.Column(pa.DateTime, nullable=False),
    })

    schema.validate(gdf, lazy=True)

    hashlib_gdf = create_gdf_hash(gdf)
    output_path = pathlib.Path("data/temp") / f"{satellite.shortName}_{hashlib_gdf}_{chunksize}"
    output_path.mkdir(parents=True, exist_ok=True)

    existing_chunks = {int(f.stem) for f in output_path.glob("*.parquet") if f.stem.isdigit()}

    gdf = quadtree_clustering(gdf, max_size=chunksize)

    for idx, chunk in enumerate(sorted(gdf.cluster_id.unique().tolist())):
        if idx in existing_chunks:
            continue

        output_filestem = str(output_path) + "/" + f"{idx}"

        chunk_df = multiple_sits_multithread(
            gdf[gdf.cluster_id == chunk],
            satellite,
            mini_chunksize=mini_chunksize,
            num_threads_rush=initial_concurrency,
            num_threads_retry=retry_concurrency,
        )

        chunk_df.to_parquet(f"{output_filestem}.parquet")

    whole_result_df = pd.DataFrame()

    for f in output_path.glob("*.parquet"):
        chunk_id = int(f.stem)
        chunk_df = pd.read_parquet(f)
        chunk_df["chunk_id"] = chunk_id
        whole_result_df = pd.concat([whole_result_df, chunk_df], ignore_index=True)

    whole_result_df["00_indexnum"] = (
        whole_result_df["chunk_id"] * (whole_result_df["00_indexnum"].max() + 1) + whole_result_df["00_indexnum"]
    )
    whole_result_df.drop(columns=["chunk_id"], inplace=True)
    whole_result_df = whole_result_df.sort_values("00_indexnum", kind="stable").reset_index(drop=True)

    remove_underscore_in_df(whole_result_df)
    whole_result_df = long_to_wide_dataframe(whole_result_df)

    return whole_result_df


def multiple_sits_task_gdrive(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    file_stem: str,
    taskname: str = "",
    gee_save_folder: str = "GEE_EXPORTS",
) -> None:
    if taskname == "":
        taskname = file_stem

    fc = ee_gdf_to_feature_collection(gdf)
    ee_expression = ee.FeatureCollection(fc.map(satellite.compute)).flatten()

    task = ee.batch.Export.table.toDrive(
        collection=ee_expression,
        description=taskname,
        fileFormat="CSV",
        fileNamePrefix=file_stem,
        folder=gee_save_folder,
        selectors=["00_indexnum", "01_doy", *satellite.selectedBands],
    )

    task.start()


def multiple_sits_task_gcs(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    bucket_name: str,
    file_path: str,
    taskname: str = "",
) -> None:
    if taskname == "":
        taskname = file_path

    fc = ee_gdf_to_feature_collection(gdf)
    ee_expression = ee.FeatureCollection(fc.map(satellite.compute)).flatten()

    task = ee.batch.Export.table.toCloudStorage(
        bucket=bucket_name,
        collection=ee_expression,
        description=taskname,
        fileFormat="CSV",
        fileNamePrefix=file_path,
        selectors=["00_indexnum", "01_doy", *satellite.selectedBands],
    )

    task.start()


def multiple_sits_chunks_gdrive(
    gdf: gpd.GeoDataFrame, satellite: AbstractSatellite, cluster_size: int = 500, gee_save_folder: str = "GEE_EXPORTS"
) -> None:
    tasks_df = ee_get_tasks_status()
    completed_or_running_tasks = set(
        tasks_df.description.apply(lambda x: x.split("_", 1)[0] + "_" + x.split("_", 2)[2]).tolist()
    )  # The task is the same, no matter who started it

    gdf = quadtree_clustering(gdf, cluster_size)
    username = getpass.getuser().replace("_", "")
    hashname = create_gdf_hash(gdf)

    for cluster_id in tqdm(sorted(gdf.cluster_id.unique())):
        cluster_id = int(cluster_id)

        if f"agl_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}" not in completed_or_running_tasks:
            multiple_sits_task_gdrive(
                gdf[gdf.cluster_id == cluster_id],
                satellite,
                f"{satellite.shortName}_{hashname}_{cluster_id}",
                f"agl_{username}_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}",
                gee_save_folder,
            )


def multiple_sits_chunks_gcs(
    gdf: gpd.GeoDataFrame, satellite: AbstractSatellite, bucket_name: str, cluster_size: int = 500
) -> None:
    tasks_df = ee_get_tasks_status()
    completed_or_running_tasks = set(
        tasks_df.description.apply(lambda x: x.split("_", 1)[0] + "_" + x.split("_", 2)[2]).tolist()
    )  # The task is the same, no matter who started it

    gdf = quadtree_clustering(gdf, cluster_size)
    username = getpass.getuser().replace("_", "")
    hashname = create_gdf_hash(gdf)

    for cluster_id in tqdm(sorted(gdf.cluster_id.unique())):
        cluster_id = int(cluster_id)

        if f"agl_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}" not in completed_or_running_tasks:
            # TODO: Also skip if the file already exists in GCS
            multiple_sits_task_gcs(
                gdf[gdf.cluster_id == cluster_id],
                satellite,
                bucket_name,
                f"{satellite.shortName}_{hashname}/{cluster_id}",
                f"agl_{username}_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}",
            )
