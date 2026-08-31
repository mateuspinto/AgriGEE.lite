import io
import pathlib
import zipfile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response, StreamingResponse

from agrigee_lite.api._jobs import JobStatus, JobType, job_store
from agrigee_lite.api._models import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])

_MISSING_RESULT_DETAIL = (
    "Job result is unavailable (it predates result persistence, or its cached file was removed) — "
    "resubmit the request."
)


def _safe_result(job_type: JobType | None, result: object) -> object:
    """Return result for status endpoints; exclude large SITS DataFrames."""
    if job_type == JobType.SITS:
        return None
    return result


@router.get("", response_model=list[JobResponse], operation_id="list_jobs")
async def list_jobs() -> list[JobResponse]:
    """List all submitted jobs and their current status."""
    return [
        JobResponse(id=j.id, type=j.type, status=j.status, result=_safe_result(j.type, j.result), error=j.error)
        for j in job_store.all()
    ]


@router.get("/{job_id}", response_model=JobResponse, operation_id="get_job")
async def get_job(job_id: str) -> JobResponse:
    """Get status and result (when complete) for a single job.

    SITS job results are not included here — use ``GET /jobs/{job_id}/download``
    to retrieve the full time-series as a Parquet file.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse(id=job.id, type=job.type, status=job.status, result=_safe_result(job.type, job.result), error=job.error)


@router.delete("/{job_id}", status_code=204, operation_id="delete_job")
async def delete_job(job_id: str) -> None:
    """Remove a completed or failed job from the store.

    Also removes the SITS result Parquet file from disk, if any — otherwise
    it would linger in the cache forever (see sits_job_result_path).
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Cannot delete a running job.")
    if job.type == JobType.SITS and job.result:
        pathlib.Path(job.result["parquet_path"]).unlink(missing_ok=True)
    job_store.delete(job_id)


@router.get("/{job_id}/download", operation_id="download_job_result")
async def download_job_result(job_id: str) -> Response:
    """
    Download the result of a completed job.

    - **images job** → ZIP archive containing one ``.zip`` file per downloaded image date.
    - **sits job** → Parquet file with the full time-series DataFrame.

    Returns 404 if the job does not exist, 409 if it is not yet completed.
    """
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=409,
            detail=f"Job is not completed yet (status: {job.status}).",
        )

    # ------------------------------------------------------------------ images
    if job.type == JobType.IMAGES:
        if not job.result:
            raise HTTPException(status_code=404, detail=_MISSING_RESULT_DETAIL)

        cache_dir = pathlib.Path(job.result["cache_dir"])
        zip_files = sorted(cache_dir.glob("*.zip"))
        if not zip_files:
            raise HTTPException(status_code=404, detail="No image files found in cache.")

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
            for zp in zip_files:
                zf.write(zp, arcname=zp.name)
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{job_id}_images.zip"'},
        )

    # -------------------------------------------------------------------- sits
    if job.type == JobType.SITS:
        if not job.result:
            raise HTTPException(status_code=404, detail=_MISSING_RESULT_DETAIL)

        result_path = pathlib.Path(job.result["parquet_path"])
        if not result_path.exists():
            raise HTTPException(status_code=404, detail="SITS result file is missing from cache — resubmit the request.")

        return FileResponse(
            result_path,
            media_type="application/octet-stream",
            filename=f"{job_id}_sits.parquet",
        )

    raise HTTPException(status_code=400, detail="This job type does not support file download.")
