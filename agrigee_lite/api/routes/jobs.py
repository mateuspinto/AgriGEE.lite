import io
import zipfile

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse

from agrigee_lite.api._jobs import JobStatus, JobType, job_store
from agrigee_lite.api._models import JobResponse

router = APIRouter(prefix="/jobs", tags=["jobs"])


@router.get("", response_model=list[JobResponse])
async def list_jobs() -> list[JobResponse]:
    """List all submitted jobs and their current status."""
    return [JobResponse(id=j.id, type=j.type, status=j.status, result=j.result, error=j.error) for j in job_store.all()]


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str) -> JobResponse:
    """Get status and result (when complete) for a single job."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobResponse(id=job.id, type=job.type, status=job.status, result=job.result, error=job.error)


@router.delete("/{job_id}", status_code=204)
async def delete_job(job_id: str) -> None:
    """Remove a completed or failed job from the store."""
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=409, detail="Cannot delete a running job.")
    job_store.delete(job_id)


@router.get("/{job_id}/download")
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
        import pathlib

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
        df = pd.DataFrame(job.result)
        buf = io.BytesIO()
        df.to_parquet(buf, index=False, engine="pyarrow", compression="brotli")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{job_id}_sits.parquet"'},
        )

    raise HTTPException(status_code=400, detail="This job type does not support file download.")
