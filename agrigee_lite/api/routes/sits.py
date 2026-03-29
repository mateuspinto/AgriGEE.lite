import asyncio

import geopandas as gpd
import pandas as pd
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from shapely.geometry import shape

from agrigee_lite.api._jobs import JobStatus, JobType, job_store
from agrigee_lite.api._models import JobResponse, MultipleSitsRequest, SitsRequest
from agrigee_lite.api._satellites import build_satellite
from agrigee_lite.get.sits import download_multiple_sits_async, download_single_sits

router = APIRouter(prefix="/sits", tags=["sits"])


# ---------------------------------------------------------------------------
# Single geometry — synchronous (no aria2 involved); fast enough for one row
# ---------------------------------------------------------------------------


@router.post("/single", response_class=JSONResponse)
async def get_single_sits(request: SitsRequest) -> list[dict]:
    """
    Download a satellite time series for a single geometry.

    Runs synchronously in a thread pool (GEE's ``computeFeatures`` HTTP call).
    Returns the time series as a JSON array of records.
    """
    satellite = build_satellite(request.satellite.name, request.satellite.params)
    geometry = shape(request.geometry.model_dump())
    df = await asyncio.to_thread(
        download_single_sits,
        geometry,
        request.start_date,
        request.end_date,
        satellite,
        set(request.reducers) if request.reducers else None,
        request.subsampling_max_pixels,
    )
    return df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Multiple geometries — async (uses aria2); long-running → background job
# ---------------------------------------------------------------------------


async def _run_multiple_sits_job(job_id: str, request: MultipleSitsRequest) -> None:
    job = job_store.get(job_id)
    job.status = JobStatus.RUNNING
    try:
        satellite = build_satellite(request.satellite.name, request.satellite.params)
        gdf = gpd.GeoDataFrame.from_features(request.feature_collection.features)
        gdf[request.start_date_column] = pd.to_datetime(gdf[request.start_date_column])
        gdf[request.end_date_column] = pd.to_datetime(gdf[request.end_date_column])

        df = await download_multiple_sits_async(
            gdf=gdf,
            satellite=satellite,
            reducers=set(request.reducers) if request.reducers else None,
            original_index_column_name=request.original_index_column,
            start_date_column_name=request.start_date_column,
            end_date_column_name=request.end_date_column,
            subsampling_max_pixels=request.subsampling_max_pixels,
            chunksize=request.chunksize,
            max_parallel_downloads=request.max_parallel_downloads,
            max_retries_per_chunk=request.max_retries_per_chunk,
            force_redownload=request.force_redownload,
        )
        job.result = df.to_dict(orient="records")
        job.status = JobStatus.COMPLETED
    except Exception as exc:
        job.error = str(exc)
        job.status = JobStatus.FAILED


@router.post("/multiple", response_class=JSONResponse, status_code=202)
async def submit_multiple_sits_job(request: MultipleSitsRequest) -> JobResponse:
    """
    Submit a multi-geometry SITS download job.

    Returns **202 Accepted** immediately with a ``job_id``.
    Poll ``GET /jobs/{job_id}`` to track progress and retrieve the result.

    The input GeoDataFrame is encoded as a GeoJSON FeatureCollection.
    Each Feature must carry ``start_date`` and ``end_date`` properties.
    """
    job = job_store.create(JobType.SITS)
    asyncio.create_task(_run_multiple_sits_job(job.id, request))
    return JobResponse(id=job.id, type=job.type, status=job.status)
