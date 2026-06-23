import asyncio
import io
import json

import geopandas as gpd
import pandas as pd
import polars as pl
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from shapely.geometry import shape

from agrigee_lite.api._jobs import JobStatus, JobType, job_store
from agrigee_lite.api._models import JobResponse, MultipleSitsFileParams, MultipleSitsRequest, SitsRequest
from agrigee_lite.api._satellites import build_satellite
from agrigee_lite.config import ASYNC_MAX_PARALLEL_DOWNLOADS, ASYNC_MAX_RETRIES_PER_CHUNK, SITS_CHUNKSIZE
from agrigee_lite.get.sits import download_multiple_sits_async, download_single_sits
from agrigee_lite.misc import create_gdf_hash

router = APIRouter(prefix="/sits", tags=["sits"])


def _sits_to_columnar(df: pl.DataFrame) -> dict[str, list]:
    """Transpose SITS DataFrame to column-oriented format for API responses.

    Timestamps are formatted as YYYY-MM-DD strings; float band values are
    rounded to 4 decimal places.
    """
    out = df.clone()

    if "timestamp" in out.columns:
        out = out.with_columns(pl.col("timestamp").dt.strftime("%Y-%m-%d"))

    float_cols = [c for c, t in zip(out.columns, out.dtypes) if t in (pl.Float32, pl.Float64)]
    if float_cols:
        out = out.with_columns([pl.col(c).round(4) for c in float_cols])

    return out.to_dict(as_series=False)


# ---------------------------------------------------------------------------
# Single geometry — synchronous; fast enough for one row
# ---------------------------------------------------------------------------


@router.post("/single", response_class=JSONResponse)
async def get_single_sits(request: SitsRequest) -> dict[str, list]:
    """
    Download a satellite time series for a single geometry.

    Runs synchronously in a thread pool (GEE's ``computeFeatures`` HTTP call).
    Returns the time series as a column-oriented JSON object: each key is a
    band/field name and its value is an array of observations in time order.
    Timestamps are formatted as ``YYYY-MM-DD``; band values are rounded to 4
    decimal places.
    """
    satellite = build_satellite(request.satellite.name, request.satellite.params)
    geometry = shape(request.geometry.model_dump())
    try:
        df = await asyncio.to_thread(
            download_single_sits,
            geometry,
            request.start_date,
            request.end_date,
            satellite,
            set(request.reducers) if request.reducers else None,
            request.subsampling_max_pixels,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _sits_to_columnar(df)


# ---------------------------------------------------------------------------
# Multiple geometries — async (aiohttp); long-running → background job
# ---------------------------------------------------------------------------


async def _run_multiple_sits_job_core(
    job_id: str,
    gdf: gpd.GeoDataFrame,
    satellite_name: str,
    satellite_params: dict,
    reducers: list[str] | None,
    start_date_column: str,
    end_date_column: str,
    original_index_column: str,
    subsampling_max_pixels: float,
    chunksize: int,
    max_parallel_downloads: int,
    max_retries_per_chunk: int,
    force_redownload: bool,
    crs: str | None = None,
) -> None:
    job_store.update_status(job_id, JobStatus.RUNNING)
    try:
        satellite = build_satellite(satellite_name, satellite_params)
        gdf[start_date_column] = pd.to_datetime(gdf[start_date_column])
        gdf[end_date_column] = pd.to_datetime(gdf[end_date_column])
        df = await download_multiple_sits_async(
            gdf=gdf,
            satellite=satellite,
            reducers=set(reducers) if reducers else None,
            original_index_column_name=original_index_column,
            start_date_column_name=start_date_column,
            end_date_column_name=end_date_column,
            subsampling_max_pixels=subsampling_max_pixels,
            chunksize=chunksize,
            max_parallel_downloads=max_parallel_downloads,
            max_retries_per_chunk=max_retries_per_chunk,
            force_redownload=force_redownload,
            crs=crs,
        )
        job = job_store.get(job_id)
        if job is not None:
            job.result = df
        job_store.update_status(job_id, JobStatus.COMPLETED)
    except Exception as exc:
        job_store.update_status(job_id, JobStatus.FAILED, error=str(exc) or repr(exc))


async def _run_multiple_sits_job(job_id: str, request: MultipleSitsRequest) -> None:
    gdf = gpd.GeoDataFrame.from_features(request.feature_collection.features)
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    await _run_multiple_sits_job_core(
        job_id=job_id,
        gdf=gdf,
        satellite_name=request.satellite.name,
        satellite_params=request.satellite.params,
        reducers=request.reducers,
        start_date_column=request.start_date_column,
        end_date_column=request.end_date_column,
        original_index_column=request.original_index_column,
        subsampling_max_pixels=request.subsampling_max_pixels,
        chunksize=request.chunksize,
        max_parallel_downloads=request.max_parallel_downloads,
        max_retries_per_chunk=request.max_retries_per_chunk,
        force_redownload=request.force_redownload,
        crs="EPSG:4326",
    )


def _sits_job_hash(request: MultipleSitsRequest) -> str:
    import hashlib
    import json

    data = {
        "feature_collection": request.feature_collection.model_dump(),
        "satellite": request.satellite.model_dump(),
        "reducers": sorted(request.reducers) if request.reducers else None,
        "start_date_column": request.start_date_column,
        "end_date_column": request.end_date_column,
        "original_index_column": request.original_index_column,
        "subsampling_max_pixels": request.subsampling_max_pixels,
        "chunksize": request.chunksize,
        "max_parallel_downloads": request.max_parallel_downloads,
        "max_retries_per_chunk": request.max_retries_per_chunk,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()  # noqa: S324


@router.post("/multiple", response_class=JSONResponse, status_code=202)
async def submit_multiple_sits_job(request: MultipleSitsRequest) -> JobResponse:
    """
    Submit a multi-geometry SITS download job.

    Returns **202 Accepted** immediately with a ``job_id``.
    Poll ``GET /jobs/{job_id}`` to track progress and retrieve the result.

    The input GeoDataFrame is encoded as a GeoJSON FeatureCollection.
    Each Feature must carry ``start_date`` and ``end_date`` properties.

    Requests with identical parameters share the same ``job_id``. If the job
    already completed, it is returned immediately. Pass ``force_redownload=true``
    to discard the prior result and start fresh.
    """
    job_hash = _sits_job_hash(request)
    existing = job_store.get(job_hash)
    if existing is not None:
        if existing.status == JobStatus.FAILED or (request.force_redownload and existing.status == JobStatus.COMPLETED):
            job_store.delete(job_hash)
        else:
            return JobResponse(id=existing.id, type=existing.type, status=existing.status)
    job = job_store.create(JobType.SITS, job_id=job_hash)
    asyncio.create_task(_run_multiple_sits_job(job.id, request))
    return JobResponse(id=job.id, type=job.type, status=job.status)


def _sits_file_job_hash(
    gdf: gpd.GeoDataFrame,
    p: MultipleSitsFileParams,
) -> str:
    import hashlib

    gdf_hash = create_gdf_hash(gdf, p.start_date_column, p.end_date_column)
    data = {
        "gdf_hash": gdf_hash,
        "satellite": p.satellite.model_dump(),
        "reducers": sorted(p.reducers) if p.reducers else None,
        "start_date_column": p.start_date_column,
        "end_date_column": p.end_date_column,
        "original_index_column": p.original_index_column,
        "subsampling_max_pixels": p.subsampling_max_pixels,
        "chunksize": p.chunksize,
        "max_parallel_downloads": p.max_parallel_downloads,
        "max_retries_per_chunk": p.max_retries_per_chunk,
    }
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()  # noqa: S324


@router.post("/multiple/file", response_class=JSONResponse, status_code=202)
async def submit_multiple_sits_job_file(
    file: UploadFile = File(..., description="Parquet file with geometry, start_date, and end_date columns"),
    satellite: str = Form('{"name": "Sentinel2", "params": {}}', description='JSON SatelliteSpec, e.g. {"name": "Landsat8", "params": {}}'),
    reducers: str | None = Form(None, description='JSON array of reducer names, e.g. ["mean", "std"]. Omit for all reducers.'),
    start_date_column: str = Form("start_date"),
    end_date_column: str = Form("end_date"),
    original_index_column: str = Form("original_index"),
    subsampling_max_pixels: float = Form(1_000),
    chunksize: int = Form(SITS_CHUNKSIZE),
    max_parallel_downloads: int = Form(ASYNC_MAX_PARALLEL_DOWNLOADS),
    max_retries_per_chunk: int = Form(ASYNC_MAX_RETRIES_PER_CHUNK),
    force_redownload: bool = Form(False),
    crs: str = Form("EPSG:4326"),
) -> JobResponse:
    """
    Submit a multi-geometry SITS download job from a Parquet file.

    Returns **202 Accepted** immediately with a ``job_id``.
    Poll ``GET /jobs/{job_id}`` to track progress and download the result.

    The Parquet file must contain:
    - ``geometry`` — WKB geometries (standard geopandas Parquet output)
    - ``start_date`` / ``end_date`` — date columns (name overridable via form fields)
    """
    try:
        satellite_spec_dict = json.loads(satellite)
        p = MultipleSitsFileParams(
            satellite=satellite_spec_dict,
            reducers=json.loads(reducers) if (reducers and reducers.strip().startswith("[")) else None,
            start_date_column=start_date_column,
            end_date_column=end_date_column,
            original_index_column=original_index_column,
            subsampling_max_pixels=subsampling_max_pixels,
            chunksize=chunksize,
            max_parallel_downloads=max_parallel_downloads,
            max_retries_per_chunk=max_retries_per_chunk,
            force_redownload=force_redownload,
            crs=crs,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid form parameters: {exc}") from exc

    content = await file.read()
    try:
        gdf = gpd.read_parquet(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot read Parquet file: {exc}") from exc

    for col in (p.start_date_column, p.end_date_column):
        if col not in gdf.columns:
            raise HTTPException(status_code=422, detail=f"Column '{col}' not found in Parquet file")

    job_hash = _sits_file_job_hash(gdf, p)
    existing = job_store.get(job_hash)
    if existing is not None:
        if existing.status == JobStatus.FAILED or (p.force_redownload and existing.status == JobStatus.COMPLETED):
            job_store.delete(job_hash)
        else:
            return JobResponse(id=existing.id, type=existing.type, status=existing.status)

    job = job_store.create(JobType.SITS, job_id=job_hash)
    asyncio.create_task(
        _run_multiple_sits_job_core(
            job_id=job.id,
            gdf=gdf,
            satellite_name=p.satellite.name,
            satellite_params=p.satellite.params,
            reducers=p.reducers,
            start_date_column=p.start_date_column,
            end_date_column=p.end_date_column,
            original_index_column=p.original_index_column,
            subsampling_max_pixels=p.subsampling_max_pixels,
            chunksize=p.chunksize,
            max_parallel_downloads=p.max_parallel_downloads,
            max_retries_per_chunk=p.max_retries_per_chunk,
            force_redownload=p.force_redownload,
            crs=p.crs,
        )
    )
    return JobResponse(id=job.id, type=job.type, status=job.status)
