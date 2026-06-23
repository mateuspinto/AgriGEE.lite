import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from shapely.geometry import shape

from agrigee_lite.api._jobs import JobStatus, JobType, job_store
from agrigee_lite.api._models import ImagesRequest, ImagesResult, JobResponse
from agrigee_lite.api._satellites import build_satellite
from agrigee_lite.get.image import _compute_images_cache_dir, download_multiple_images_async

router = APIRouter(prefix="/images", tags=["images"])


async def _run_images_job(job_id: str, request: ImagesRequest) -> None:
    job_store.update_status(job_id, JobStatus.RUNNING)
    try:
        satellite = build_satellite(request.satellite.name, request.satellite.params)
        geometry = shape(request.geometry.model_dump())
        dates = await download_multiple_images_async(
            geometry=geometry,
            start_date=request.start_date,
            end_date=request.end_date,
            satellite=satellite,
            invalid_images_threshold=request.invalid_images_threshold,
            max_parallel_downloads=request.max_parallel_downloads,
            force_redownload=request.force_redownload,
            image_indices=request.image_indices,
        )
        from agrigee_lite.config import ASYNC_MAX_RETRIES_PER_CHUNK

        cache_dir = str(_compute_images_cache_dir(
            satellite=satellite,
            start_date=request.start_date,
            end_date=request.end_date,
            centroid_x=geometry.centroid.x,
            centroid_y=geometry.centroid.y,
            invalid_images_threshold=request.invalid_images_threshold,
            image_indices=request.image_indices,
            max_retries_per_chunk=ASYNC_MAX_RETRIES_PER_CHUNK,
            crs=None,
        ))
        job = job_store.get(job_id)
        if job is not None:
            job.result = ImagesResult(dates=dates, cache_dir=cache_dir).model_dump()
        job_store.update_status(job_id, JobStatus.COMPLETED)
    except Exception as exc:
        job_store.update_status(job_id, JobStatus.FAILED, error=str(exc))


def _images_job_hash(request: ImagesRequest) -> str:
    from agrigee_lite.config import ASYNC_MAX_RETRIES_PER_CHUNK

    satellite = build_satellite(request.satellite.name, request.satellite.params)
    geometry = shape(request.geometry.model_dump())
    return _compute_images_cache_dir(
        satellite=satellite,
        start_date=request.start_date,
        end_date=request.end_date,
        centroid_x=geometry.centroid.x,
        centroid_y=geometry.centroid.y,
        invalid_images_threshold=request.invalid_images_threshold,
        image_indices=request.image_indices,
        max_retries_per_chunk=ASYNC_MAX_RETRIES_PER_CHUNK,
        crs=None,
    ).name


@router.post("", response_class=JSONResponse, status_code=202)
async def submit_images_job(request: ImagesRequest) -> JobResponse:
    """
    Submit an image download job.

    Returns **202 Accepted** immediately with a ``job_id``.
    Poll ``GET /jobs/{job_id}`` to track progress and retrieve the result.

    Requests with identical parameters share the same ``job_id``. If the job
    already completed successfully, it is returned immediately without
    re-downloading. Pass ``force_redownload=true`` to discard the prior result
    and start fresh.
    """
    job_hash = _images_job_hash(request)
    existing = job_store.get(job_hash)
    if existing is not None:
        if existing.status == JobStatus.FAILED or (request.force_redownload and existing.status == JobStatus.COMPLETED):
            job_store.delete(job_hash)
        else:
            return JobResponse(id=existing.id, type=existing.type, status=existing.status, result=existing.result)
    job = job_store.create(JobType.IMAGES, job_id=job_hash)
    asyncio.create_task(_run_images_job(job.id, request))
    return JobResponse(id=job.id, type=job.type, status=job.status)
