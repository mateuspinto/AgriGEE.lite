import asyncio
import getpass
import json
import logging
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, cast

import aiohttp
import ee
import geopandas as gpd
import pandas as pd
import pandera.pandas as pa
import polars as pl
from shapely import MultiPolygon, Point, Polygon
from tenacity import AsyncRetrying, RetryError, stop_after_attempt, wait_random_exponential
from tqdm.auto import tqdm

from agrigee_lite._geo_compat import (
    GeoDataFrameLike,
    NormalizedGeoDataFrame,
    geometry_value_to_shapely,
    get_crs,
    normalize_geodataframe,
    transform_geometry,
    wrap_geopolars_frame,
    to_geopandas_geodataframe,
)
from agrigee_lite.cache.backend import (
    CacheEngine,
    fetch_sits_batch_coverage,
    fetch_sits_by_job_ids,
    fetch_sits_with_gaps,
    get_engine,
    store_sits_polars,
)
from agrigee_lite.config import (
    AIOHTTP_CONNECTOR_LIMIT,
    AIOHTTP_TIMEOUT_SECONDS,
    ASYNC_AIMD_INITIAL_DOWNLOADS,
    ASYNC_AIMD_SUCCESS_STRIDE,
    ASYNC_MAX_PARALLEL_DOWNLOADS,
    ASYNC_MAX_RETRIES_PER_CHUNK,
    ASYNC_MAX_URL_WORKERS,
    SITS_CHUNKSIZE,
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

logger = logging.getLogger(__name__)
TabularFrame = pd.DataFrame | pl.DataFrame


def _as_date_str(value: pd.Timestamp | str) -> str:
    return str(value)[:10]


def _wrap_normalized_geo_frame(df: pl.DataFrame, reference: NormalizedGeoDataFrame) -> NormalizedGeoDataFrame:
    return wrap_geopolars_frame(df, crs=get_crs(reference))


def _filter_normalized_geo_frame(gdf: NormalizedGeoDataFrame, predicate: pl.Expr) -> NormalizedGeoDataFrame:
    return _wrap_normalized_geo_frame(gdf.filter(predicate), gdf)


def _take_normalized_geo_rows(gdf: NormalizedGeoDataFrame, positions: list[int]) -> NormalizedGeoDataFrame:
    if not positions:
        return _wrap_normalized_geo_frame(gdf.slice(0, 0), gdf)

    with_index = gdf.with_row_index("_row_nr")
    filtered = with_index.filter(pl.col("_row_nr").is_in(positions)).sort("_row_nr").drop("_row_nr")
    return _wrap_normalized_geo_frame(filtered, gdf)


def _unique_ints(series: pl.Series) -> list[int]:
    return [int(value) for value in series.unique().drop_nulls().to_list()]


def build_ee_expression(
    gdf: GeoDataFrameLike,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    original_index_column_name: str,
    crs: str | None = None,
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
) -> ee.FeatureCollection:
    """
    Build Earth Engine expression for satellite time series computation.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopolars.GeoDataFrame
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
    effective_crs = crs or get_crs(gdf)
    fc = ee_gdf_to_feature_collection(
        gdf,
        original_index_column_name,
        start_date_column_name,
        end_date_column_name,
        crs=effective_crs,
    )
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


def prepare_output_df(
    df: TabularFrame,
    satellite: AbstractSatellite,
    original_index_column_name: str,
) -> pl.DataFrame:
    """
    Prepare and clean output DataFrame from satellite time series data.

    Parameters
    ----------
    df : pandas.DataFrame or polars.DataFrame
        Raw DataFrame from satellite time series computation.
    satellite : AbstractSatellite
        Satellite configuration object used for data processing.
    original_index_column_name : str
        Name of the column to restore original indices.

    Returns
    -------
    polars.DataFrame
        Cleaned and processed DataFrame with proper column names and data types.
    """
    out = pl.from_pandas(df) if isinstance(df, pd.DataFrame) else df.clone()

    if "geo" in out.columns:
        out = out.drop("geo")

    rename_map = {column: column.split("_", 1)[1] if "_" in column else column for column in out.columns}
    out = out.rename(rename_map)

    if isinstance(satellite, OpticalSatellite):  # Zero values in optical bands are invalid
        band_columns = sorted(set(out.columns) - {"timestamp", "validPixelsCount", "indexnum"})
        if band_columns:
            out = out.filter(~pl.all_horizontal(pl.col(c) == 0 for c in band_columns))

    if "timestamp" in out.columns:
        ts_dtype = out.schema["timestamp"]
        if ts_dtype == pl.String:
            out = out.with_columns(pl.col("timestamp").str.to_datetime(strict=False))
        elif ts_dtype == pl.Date:
            out = out.with_columns(pl.col("timestamp").cast(pl.Datetime))
        elif ts_dtype != pl.Datetime:
            out = out.with_columns(pl.col("timestamp").cast(pl.Datetime, strict=False))

        if out.get_column("timestamp").null_count() == out.height:
            out = out.drop("timestamp")

    if "indexnum" in out.columns:
        indexnum_col = out.get_column("indexnum")
        normalized_indexnum = indexnum_col.cast(pl.Int64, strict=False).fill_null(0)
        if out.height == 0 or bool((normalized_indexnum == 0).all()):
            out = out.drop("indexnum")
        else:
            out = out.sort("indexnum")

    if "indexnum" in out.columns:
        out = out.rename({"indexnum": original_index_column_name})

    return out


def sanitize_and_prepare_input_gdf(
    gdf: GeoDataFrameLike,
    satellite: AbstractSatellite,
    original_index_column_name: str,
    crs: str | None = None,
    coarse_resolution: int = 5,
    fine_resolution: int = 8,
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
) -> NormalizedGeoDataFrame:
    """
    Sanitize and prepare input GeoDataFrame for satellite time series processing.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopolars.GeoDataFrame
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
    geopolars.GeoDataFrame
        Sanitized GeoDataFrame with clustering applied and invalid data filtered.
    """
    boundary_gdf = normalize_geodataframe(gdf, crs=crs).to_geopandas()

    if original_index_column_name == "original_index":
        boundary_gdf = boundary_gdf.reset_index().rename(columns={"index": original_index_column_name})
        logger.debug("Column '%s' created to store original index.", original_index_column_name)

    schema = pa.DataFrameSchema(
        {
            "geometry": pa.Column("geometry", nullable=False),
            start_date_column_name: pa.Column(pa.DateTime, nullable=False),
            end_date_column_name: pa.Column(pa.DateTime, nullable=False),
            original_index_column_name: pa.Column(boundary_gdf[original_index_column_name].dtype),
        },
        unique=[original_index_column_name],
    )
    schema.validate(boundary_gdf, lazy=True)

    boundary_gdf = boundary_gdf[
        ["geometry", start_date_column_name, end_date_column_name, original_index_column_name]
    ].copy()
    normalized = normalize_geodataframe(boundary_gdf, crs=crs)
    normalized = _wrap_normalized_geo_frame(
        normalized.with_columns(
            pl.col(start_date_column_name).cast(pl.Datetime, strict=False),
            pl.col(end_date_column_name).cast(pl.Datetime, strict=False),
        ),
        normalized,
    )

    sat_start = pd.Timestamp(satellite.startDate)
    sat_end = pd.Timestamp(satellite.endDate)
    prepared = _wrap_normalized_geo_frame(
        normalized.with_columns(
            (
                (pl.col(end_date_column_name) < pl.lit(sat_start)) | (pl.col(start_date_column_name) > pl.lit(sat_end))
            ).alias("_mask_no_intersection"),
            (
                (pl.col(start_date_column_name) >= pl.lit(sat_start))
                & (pl.col(end_date_column_name) <= pl.lit(sat_end))
            ).alias("_mask_total_intersection"),
        ),
        normalized,
    )

    count_none = int(prepared.get_column("_mask_no_intersection").sum())
    count_total = int(prepared.get_column("_mask_total_intersection").sum())
    count_partial = prepared.height - count_none - count_total

    pct_none = 100 * count_none / prepared.height
    if pct_none > 0:
        logger.debug("%.2f%% of the data do not intersect the satellite period.", pct_none)

    pct_partial = 100 * count_partial / prepared.height
    if pct_partial > 0:
        logger.debug("%.2f%% of the data partially intersect the satellite period.", pct_partial)

    filtered = _wrap_normalized_geo_frame(
        _filter_normalized_geo_frame(prepared, ~pl.col("_mask_no_intersection")).drop(
            "_mask_no_intersection", "_mask_total_intersection"
        ),
        prepared,
    )

    if filtered.height == 0:
        return _wrap_normalized_geo_frame(filtered, filtered)

    clustered = cast(
        NormalizedGeoDataFrame,
        h3_clustering(filtered, coarse_resolution=coarse_resolution, fine_resolution=fine_resolution, crs=crs),
    )
    return clustered


def download_single_sits(
    geometry: Polygon | MultiPolygon | Point,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    reducers: set[str] | None = None,
    subsampling_max_pixels: float = 1_000,
    crs: str | None = None,
) -> pl.DataFrame:
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
    polars.DataFrame
        DataFrame containing satellite time series data.

    Raises
    ------
    ValueError
        If the requested period does not intersect with satellite's temporal range.
    """
    start_date = _as_date_str(start_date)
    end_date = _as_date_str(end_date)

    if end_date < satellite.startDate or start_date > satellite.endDate:
        raise ValueError(  # noqa: TRY003
            f"Requested period ({start_date} to {end_date}) does not intersect with satellite's range "
            f"({satellite.startDate} to {satellite.endDate})"
        )

    geometry_wgs84 = transform_geometry(geometry, crs)

    _engine = get_engine()
    cached_df = pl.DataFrame()
    gaps: list[tuple[str, str]] = [(start_date, end_date)]

    if _engine is not None:
        cached_df, gaps = fetch_sits_with_gaps(
            _engine, geometry_wgs84, start_date, end_date, satellite, reducers, subsampling_max_pixels
        )
        if not gaps:
            logger.debug("Cache hit: %s %s→%s", satellite.shortName, start_date, end_date)
            return cached_df

    gap_dfs: list[pl.DataFrame] = []
    for gap_start, gap_end in gaps:
        ee_feature = ee.Feature(
            geometry_wgs84.__geo_interface__,
            {"s": gap_start, "e": gap_end, "0": 0},
        )
        ee_expression = satellite.compute(ee_feature, reducers=reducers, subsampling_max_pixels=subsampling_max_pixels)
        gap_raw = ee.data.computeFeatures({"expression": ee_expression, "fileFormat": "PANDAS_DATAFRAME"})
        gap_df = prepare_output_df(gap_raw, satellite, "IGNORED")
        if _engine is not None:
            store_sits_polars(
                _engine, gap_df, geometry_wgs84, gap_start, gap_end, satellite, reducers, subsampling_max_pixels
            )
        if not gap_df.is_empty():
            gap_dfs.append(gap_df)

    all_new = pl.concat(gap_dfs, rechunk=False) if gap_dfs else pl.DataFrame()

    if cached_df.is_empty():
        return all_new
    if all_new.is_empty():
        return cached_df

    result = pl.concat([cached_df, all_new], rechunk=False)
    if "timestamp" in result.columns:
        result = result.sort("timestamp")
    return result


class _AdaptiveSemaphore:
    """AIMD concurrency limiter: +1 every `success_stride` successes, //2 on rate-limit."""

    def __init__(self, initial: int, minimum: int, maximum: int, success_stride: int = 1) -> None:
        self._limit = initial
        self._active = 0
        self._minimum = minimum
        self._maximum = maximum
        self._success_stride = success_stride
        self._success_count = 0
        self._cond = asyncio.Condition()

    async def __aenter__(self) -> "_AdaptiveSemaphore":
        async with self._cond:
            while self._active >= self._limit:
                await self._cond.wait()
            self._active += 1
        return self

    async def __aexit__(self, *_: object) -> None:
        async with self._cond:
            self._active -= 1
            self._cond.notify()

    async def on_success(self) -> None:
        async with self._cond:
            self._success_count += 1
            if self._success_count % self._success_stride == 0 and self._limit < self._maximum:
                self._limit += 1
                self._cond.notify()

    async def on_rate_limit(self) -> None:
        async with self._cond:
            self._limit = max(self._minimum, self._limit // 2)

    @property
    def limit(self) -> int:
        return self._limit


def _is_429(exc: BaseException) -> bool:
    if isinstance(exc, aiohttp.ClientResponseError) and exc.status == 429:
        return True
    if isinstance(exc, ee.EEException):
        msg = str(exc)
        return "429" in msg or "quota exceeded" in msg.lower()
    return False


def _store_chunk(
    engine: CacheEngine,
    chunk_pl: pl.DataFrame,
    sub_gdf: NormalizedGeoDataFrame,
    satellite: AbstractSatellite,
    reducers: set[str] | None,
    subsampling_max_pixels: float,
    original_index_col: str,
    start_date_col: str,
    end_date_col: str,
) -> None:
    if chunk_pl.is_empty():
        return

    rename_map = {
        column: (
            original_index_col
            if column == "00_indexnum"
            else "timestamp"
            if column == "01_timestamp"
            else column.split("_", 1)[1]
        )
        for column in chunk_pl.columns
        if "_" in column
    }
    prepared_pl = chunk_pl.drop("geo") if "geo" in chunk_pl.columns else chunk_pl
    prepared_pl = prepared_pl.rename(rename_map)
    if original_index_col not in prepared_pl.columns:
        raise KeyError(f"Chunk is missing the expected {original_index_col!r} column.")

    for original_index in prepared_pl.get_column(original_index_col).unique(maintain_order=True).to_list():
        feat_pl = prepared_pl.filter(pl.col(original_index_col) == original_index)
        matches = _filter_normalized_geo_frame(sub_gdf, pl.col(original_index_col) == pl.lit(original_index))
        if matches.height == 0:
            raise KeyError(f"Could not match chunk row for {original_index_col}={original_index!r}.")
        row = matches.row(0, named=True)
        store_sits_polars(
            engine,
            feat_pl,
            geometry_value_to_shapely(cast(bytes | bytearray | memoryview, row["geometry"])),
            _as_date_str(row[start_date_col]),
            _as_date_str(row[end_date_col]),
            satellite,
            reducers,
            subsampling_max_pixels,
        )


async def download_multiple_sits_async(  # noqa: C901
    gdf: GeoDataFrameLike,
    satellite: AbstractSatellite,
    reducers: set[str] | None = None,
    original_index_column_name: str = "original_index",
    crs: str | None = None,
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
    subsampling_max_pixels: float = 1_000,
    chunksize: int = SITS_CHUNKSIZE,
    max_parallel_downloads: int = ASYNC_MAX_PARALLEL_DOWNLOADS,
    max_retries_per_chunk: int = ASYNC_MAX_RETRIES_PER_CHUNK,
    force_redownload: bool = False,
) -> pl.DataFrame:
    if len(gdf) == 0:
        return pl.DataFrame()

    prepared_gdf = sanitize_and_prepare_input_gdf(
        gdf,
        satellite,
        original_index_column_name,
        crs=crs,
        start_date_column_name=start_date_column_name,
        end_date_column_name=end_date_column_name,
    )

    if prepared_gdf.height == 0:
        return pl.DataFrame()

    _engine = get_engine()
    if _engine is None:
        raise RuntimeError

    cached_items: list[tuple[list[int], Any, str, str]] = []
    uncached_positions: list[int] = []

    if not force_redownload:
        batch_coverage = fetch_sits_batch_coverage(
            _engine,
            prepared_gdf,
            satellite,
            reducers,
            subsampling_max_pixels,
            start_date_column_name,
            end_date_column_name,
            crs,
        )
        for pos in range(prepared_gdf.height):
            coverage = batch_coverage.get(pos)
            if coverage is not None:
                job_ids, gaps = coverage
                if not gaps:
                    row = prepared_gdf.row(pos, named=True)
                    cached_items.append((
                        job_ids,
                        row[original_index_column_name],
                        str(row[start_date_column_name])[:10],
                        str(row[end_date_column_name])[:10],
                    ))
                    continue
            uncached_positions.append(pos)
    else:
        uncached_positions = list(range(prepared_gdf.height))

    uncached_request_rows = _take_normalized_geo_rows(prepared_gdf, uncached_positions)

    def _finalize_from_cache() -> pl.DataFrame:
        if not cached_items:
            return pl.DataFrame()

        all_job_ids = list({jid for jids, _, _, _ in cached_items for jid in jids})
        cached_data = fetch_sits_by_job_ids(_engine, satellite, all_job_ids)

        pl_frames: list[pl.DataFrame] = []
        for job_ids, orig_idx, q_start, q_end in cached_items:
            sub_dfs = [cached_data[jid] for jid in job_ids if jid in cached_data]
            if not sub_dfs:
                continue
            combined = pl.concat(sub_dfs, rechunk=False).unique(subset=["timestamp"], maintain_order=True)
            ts_start = pd.Timestamp(q_start).to_pydatetime()
            ts_end = (pd.Timestamp(q_end) + pd.Timedelta(days=1)).to_pydatetime()
            filtered = combined.filter(
                (pl.col("timestamp") >= pl.lit(ts_start)) & (pl.col("timestamp") < pl.lit(ts_end))
            )
            if not filtered.is_empty():
                pl_frames.append(filtered.with_columns(pl.lit(orig_idx).alias(original_index_column_name)))

        if not pl_frames:
            return pl.DataFrame()

        return pl.concat(pl_frames, rechunk=False).sort([original_index_column_name, "timestamp"])

    if uncached_request_rows.height == 0:
        return _finalize_from_cache()

    num_chunks = (uncached_request_rows.height + chunksize - 1) // chunksize

    selectors = build_selectors(satellite, reducers)
    semaphore = _AdaptiveSemaphore(
        initial=min(ASYNC_AIMD_INITIAL_DOWNLOADS, max_parallel_downloads),
        minimum=1,
        maximum=max_parallel_downloads,
        success_stride=ASYNC_AIMD_SUCCESS_STRIDE,
    )
    url_semaphore = asyncio.Semaphore(ASYNC_MAX_URL_WORKERS)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=ASYNC_MAX_URL_WORKERS, thread_name_prefix="agrigee_gee")
    _is_tty = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    pbar = tqdm(
        total=num_chunks,
        unit="chunk",
        smoothing=0,
        mininterval=0.3 if _is_tty else 30.0,
        dynamic_ncols=True,
        bar_format="{percentage:3.0f}% | {n_fmt}/{total_fmt} | [{elapsed}<{remaining}, {rate_fmt}] | {postfix}",
    )

    stats: dict[str, int] = {
        "cache": len(cached_items),
        "done": 0,
        "ok": 0,
        "err": 0,
        "retry": 0,
    }

    def _update_postfix() -> None:
        pbar.set_postfix_str(
            f"d:{stats['done']} ok:{stats['ok']} e:{stats['err']} r:{stats['retry']} c:{stats['cache']} lim:{semaphore.limit}",
            refresh=False,
        )

    _update_postfix()

    _non_band_raw = {"00_indexnum", "01_timestamp", "99_validPixelsCount"}
    store_queue: asyncio.Queue[tuple[pl.DataFrame, NormalizedGeoDataFrame] | None] = asyncio.Queue()

    async def store_consumer() -> None:
        loop_ = asyncio.get_running_loop()
        while True:
            item = await store_queue.get()
            if item is None:
                break
            chunk_pl, sub_gdf = item
            await loop_.run_in_executor(
                None,
                _store_chunk,
                _engine,
                chunk_pl,
                sub_gdf,
                satellite,
                reducers,
                subsampling_max_pixels,
                original_index_column_name,
                start_date_column_name,
                end_date_column_name,
            )

    async def fetch_chunk(
        session: aiohttp.ClientSession,
        chunk_id: int,
        sub: NormalizedGeoDataFrame,
    ) -> pl.DataFrame:
        def get_url() -> str:
            expr = build_ee_expression(
                sub,
                satellite,
                reducers,
                subsampling_max_pixels,
                original_index_column_name,
                crs,
                start_date_column_name,
                end_date_column_name,
            )
            return expr.getDownloadURL(filetype="csv", selectors=selectors, filename=str(chunk_id))

        async with url_semaphore:
            url = await loop.run_in_executor(executor, get_url)

        async with session.get(url, timeout=aiohttp.ClientTimeout(total=AIOHTTP_TIMEOUT_SECONDS)) as resp:
            resp.raise_for_status()
            data = await resp.read()

        frame = pl.read_csv(data)
        if "geo" in frame.columns:
            frame = frame.drop("geo")
        if isinstance(satellite, OpticalSatellite):
            band_cols = [c for c in frame.columns if c not in _non_band_raw]
            if band_cols:
                frame = frame.filter(~pl.all_horizontal(pl.col(c) == 0 for c in band_cols))
        return frame

    async def fetch_with_retry(session: aiohttp.ClientSession, chunk_id: int) -> pl.DataFrame:
        async with semaphore:
            start = chunk_id * chunksize
            positions = list(range(start, min(start + chunksize, uncached_request_rows.height)))
            sub = _take_normalized_geo_rows(uncached_request_rows, positions)
            chunk_df: pl.DataFrame = pl.DataFrame()
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(max_retries_per_chunk),
                    wait=wait_random_exponential(multiplier=2, min=4, max=60),
                ):
                    if attempt.retry_state.attempt_number > 1:
                        stats["retry"] += 1
                        prev_exc = attempt.retry_state.outcome.exception() if attempt.retry_state.outcome else None
                        if prev_exc is not None and _is_429(prev_exc):
                            await semaphore.on_rate_limit()
                        _update_postfix()
                    with attempt:
                        chunk_df = await fetch_chunk(session, chunk_id, sub)
                await semaphore.on_success()
                stats["ok"] += 1
            except RetryError:
                stats["err"] += 1
                logger.debug("Chunk %d failed after %d attempts.", chunk_id, max_retries_per_chunk, exc_info=True)
                return pl.DataFrame()
            except Exception:
                stats["err"] += 1
                logger.debug("Chunk %d failed with unexpected error.", chunk_id, exc_info=True)
                return pl.DataFrame()
            else:
                await store_queue.put((chunk_df, sub))
                return chunk_df
            finally:
                stats["done"] += 1
                _update_postfix()
                pbar.update(1)

    def _cancel_all() -> None:
        logger.warning("Download interrupted. Exiting.")
        raise SystemExit(1)

    _signals_registered = False
    try:
        loop.add_signal_handler(signal.SIGINT, _cancel_all)
        loop.add_signal_handler(signal.SIGTERM, _cancel_all)
        _signals_registered = True
    except (NotImplementedError, ValueError):
        pass

    connector = aiohttp.TCPConnector(limit=max(max_parallel_downloads, AIOHTTP_CONNECTOR_LIMIT))
    consumer_task = asyncio.create_task(store_consumer())
    results: list[pl.DataFrame | BaseException] = []
    try:
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [asyncio.create_task(fetch_with_retry(session, cid)) for cid in range(num_chunks)]
            results = cast(list[pl.DataFrame | BaseException], await asyncio.gather(*tasks, return_exceptions=True))
    finally:
        await store_queue.put(None)
        try:
            await consumer_task
        finally:
            executor.shutdown(wait=False, cancel_futures=True)
            if _signals_registered:
                loop.remove_signal_handler(signal.SIGINT)
                loop.remove_signal_handler(signal.SIGTERM)
            pbar.close()

    pl_frames: list[pl.DataFrame] = []
    for cid, res in enumerate(results):
        if isinstance(res, BaseException):
            logger.debug("Chunk %d raised: %s", cid, res)
        elif res.height > 0:
            pl_frames.append(res)

    raw_df = pl.concat(pl_frames, rechunk=False) if pl_frames else pl.DataFrame()
    whole_result_df = prepare_output_df(raw_df, satellite, original_index_column_name)
    cached_df = _finalize_from_cache()
    if cached_df.is_empty():
        return whole_result_df
    if whole_result_df.is_empty():
        return cached_df

    result_df = pl.concat([cached_df, whole_result_df], rechunk=False)
    if "timestamp" in result_df.columns:
        result_df = result_df.sort([original_index_column_name, "timestamp"])
    return result_df


def download_multiple_sits_chunks_gdrive(
    gdf: GeoDataFrameLike,
    satellite: AbstractSatellite,
    reducers: set[str] | None = None,
    original_index_column_name: str = "original_index",
    crs: str | None = None,
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
    subsampling_max_pixels: float = 1_000,
    coarse_resolution: int = 5,
    fine_resolution: int = 8,
    gee_save_folder: str = "AGL_EXPORTS",
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
    wait : bool, optional
        Whether to wait for task completion, by default True.
    """
    if len(gdf) == 0:
        return None

    def download_multiple_sits_task_gdrive(
        gdf: GeoDataFrameLike,
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
            crs=crs,
            start_date_column_name=start_date_column_name,
            end_date_column_name=end_date_column_name,
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

    prepared_gdf = sanitize_and_prepare_input_gdf(
        gdf,
        satellite,
        original_index_column_name,
        crs=crs,
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
    hashname = create_gdf_hash(prepared_gdf, start_date_column_name, end_date_column_name)
    cluster_ids = sorted(_unique_ints(prepared_gdf.get_column("cluster_id")))

    for cluster_id in tqdm(
        cluster_ids, desc=f"Creating GEE tasks ({satellite.shortName}_{hashname}_r{coarse_resolution})"
    ):
        if f"agl_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}" not in completed_or_running_tasks:
            cluster_gdf = _filter_normalized_geo_frame(prepared_gdf, pl.col("cluster_id") == pl.lit(cluster_id))
            task = download_multiple_sits_task_gdrive(
                cluster_gdf,
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
    gdf: GeoDataFrameLike,
    satellite: AbstractSatellite,
    bucket_name: str,
    reducers: set[str] | None = None,
    original_index_column_name: str = "original_index",
    crs: str | None = None,
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
    subsampling_max_pixels: float = 1_000,
    coarse_resolution: int = 5,
    fine_resolution: int = 8,
    wait: bool = True,
) -> None | pl.DataFrame:
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
    wait : bool, optional
        Whether to wait for task completion, by default True.

    Returns
    -------
    None or polars.DataFrame
        If wait is True, returns DataFrame with combined results.
        If wait is False, returns None.
    """
    from smart_open import open as smart_open  # pyright: ignore[reportMissingImports]

    if len(gdf) == 0:
        logging.warning("Empty GeoDataFrame, nothing to download")
        return None

    def download_multiple_sits_task_gcs(
        gdf: GeoDataFrameLike,
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
            crs=crs,
            start_date_column_name=start_date_column_name,
            end_date_column_name=end_date_column_name,
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

    prepared_gdf = sanitize_and_prepare_input_gdf(
        gdf,
        satellite,
        original_index_column_name,
        crs=crs,
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
    hashname = create_gdf_hash(prepared_gdf, start_date_column_name, end_date_column_name)

    gcs_save_folder = f"agl/{satellite.shortName}_{hashname}"
    metadata_dict: dict[str, Any] = {}
    metadata_dict |= log_dict_function_call_summary(["gdf", "satellite"])
    metadata_dict |= satellite.log_dict()
    metadata_dict["user"] = username
    metadata_dict["creation_date"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    with smart_open(f"gs://{bucket_name}/{gcs_save_folder}/metadata.json", "w") as f:
        json.dump(metadata_dict, f, indent=4)

    with smart_open(f"gs://{bucket_name}/{gcs_save_folder}/geodataframe.parquet", "wb") as f:
        to_geopandas_geodataframe(prepared_gdf).to_parquet(f, compression="brotli")

    file_uris = []
    cluster_ids = sorted(_unique_ints(prepared_gdf.get_column("cluster_id")))

    for cluster_id in tqdm(cluster_ids):
        if f"agl_multiple_sits_{satellite.shortName}_{hashname}_{cluster_id}" not in completed_or_running_tasks:
            # TODO: Also skip if the file already exists in GCS
            cluster_gdf = _filter_normalized_geo_frame(prepared_gdf, pl.col("cluster_id") == pl.lit(cluster_id))
            task = download_multiple_sits_task_gcs(
                cluster_gdf,
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

        dfs: list[pl.DataFrame] = []
        for file_uri in file_uris:
            with smart_open(file_uri, "rb") as f:
                dfs.append(pl.read_csv(f))

        df = pl.concat(dfs, rechunk=False) if dfs else pl.DataFrame()
        df = prepare_output_df(df, satellite, original_index_column_name)
        return df
    else:
        return None
