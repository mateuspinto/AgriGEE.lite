"""
DB-backed job store for tracking long-running download tasks.

Jobs are persisted to DuckDB/PostGIS (whichever backend is active).
An in-memory dict provides O(1) reads without hitting the DB on every poll.

On startup, call ``job_store.load_from_db()`` after the cache is initialized
to restore jobs that survived a server restart. Jobs that were RUNNING when
the server died are reset to FAILED.
"""

import enum
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agrigee_lite.cache.backend import (
    create_api_job,
    delete_api_job,
    get_engine,
    list_api_jobs,
    update_api_job,
)


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobType(str, enum.Enum):
    IMAGES = "images"
    SITS = "sits"


@dataclass
class Job:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: JobType | None = None
    status: JobStatus = JobStatus.PENDING
    result: Any = None
    error: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    """
    Write-through job store: mutations persist to DB immediately; reads come
    from memory. Call ``load_from_db()`` once after cache initialization to
    hydrate the in-memory dict from prior runs.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def load_from_db(self) -> None:
        """Load persisted jobs from DB. Reset orphaned RUNNING jobs to FAILED.

        Safe to call on a DB that predates the api_jobs table — the table is
        created automatically if missing.
        """
        from agrigee_lite.cache.backend import ensure_api_jobs_table

        engine = get_engine()
        if engine is None:
            return
        ensure_api_jobs_table(engine)
        now = _now()
        for row in list_api_jobs(engine):
            status = JobStatus(row["status"])
            if status == JobStatus.RUNNING:
                status = JobStatus.FAILED
                error = "server restarted while job was running"
                update_api_job(engine, row["id"], status.value, error, now)
            else:
                error = row["error"]
            job = Job(
                id=row["id"],
                type=JobType(row["type"]) if row["type"] else None,
                status=status,
                error=error,
            )
            self._jobs[job.id] = job

    def create(self, job_type: JobType | None = None, job_id: str | None = None) -> Job:
        job = Job(id=job_id or str(uuid.uuid4()), type=job_type)
        self._jobs[job.id] = job
        engine = get_engine()
        if engine is not None:
            create_api_job(
                engine,
                job.id,
                job_type.value if job_type else None,
                job.status.value,
                _now(),
            )
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def update_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job.status = status
        job.error = error
        engine = get_engine()
        if engine is not None:
            update_api_job(engine, job_id, status.value, error, _now())

    def delete(self, job_id: str) -> bool:
        existed = self._jobs.pop(job_id, None) is not None
        if existed:
            engine = get_engine()
            if engine is not None:
                delete_api_job(engine, job_id)
        return existed


job_store = JobStore()
