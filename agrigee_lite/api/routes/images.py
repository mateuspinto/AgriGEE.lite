import asyncio
import pathlib

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from shapely.geometry import shape

from agrigee_lite.api._jobs import JobStatus, JobType, job_store
from agrigee_lite.api._models import ImagesRequest, ImagesResult, JobResponse
from agrigee_lite.api._satellites import build_satellite
from agrigee_lite.get.image import download_multiple_images_async

router = APIRouter(prefix="/images", tags=["images"])


async def _run_images_job(job_id: str, request: ImagesRequest) -> None:
    job = job_store.get(job_id)
    job.status = JobStatus.RUNNING
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
        # Reconstruct the cache path to inform the caller where files landed
        from agrigee_lite.misc import create_dict_hash, log_dict_function_call_summary

        metadata_dict: dict = {}
        metadata_dict |= log_dict_function_call_summary([
            "geometry",
            "start_date",
            "end_date",
            "satellite",
            "max_parallel_downloads",
            "force_redownload",
        ])
        metadata_dict |= satellite.log_dict()
        metadata_dict["start_date"] = request.start_date
        metadata_dict["end_date"] = request.end_date
        metadata_dict["centroid_x"] = geometry.centroid.x
        metadata_dict["centroid_y"] = geometry.centroid.y
        cache_dir = str(pathlib.Path.home() / ".cache" / "agrigee_lite" / "images" / create_dict_hash(metadata_dict))
        job.result = ImagesResult(dates=dates, cache_dir=cache_dir).model_dump()
        job.status = JobStatus.COMPLETED
    except Exception as exc:
        job.error = str(exc)
        job.status = JobStatus.FAILED


@router.post("", response_class=JSONResponse, status_code=202)
async def submit_images_job(request: ImagesRequest) -> JobResponse:
    """
    Submit an image download job.

    Returns **202 Accepted** immediately with a ``job_id``.
    Poll ``GET /jobs/{job_id}`` to track progress and retrieve the result.
    """
    job = job_store.create(JobType.IMAGES)
    asyncio.create_task(_run_images_job(job.id, request))
    return JobResponse(id=job.id, type=job.type, status=job.status)
