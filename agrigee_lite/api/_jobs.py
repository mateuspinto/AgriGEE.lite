"""
In-memory job store for tracking long-running download tasks.

For multi-process deployments, replace JobStore with a Redis-backed
implementation that implements the same interface.
"""

import enum
import uuid
from dataclasses import dataclass, field
from typing import Any


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


class JobStore:
    """
    Single-process in-memory store.

    Thread/coroutine safety: all mutations happen on the asyncio event loop
    (job runners are coroutines), so no locking is needed. If you move to
    threads or multi-process, wrap mutations with asyncio.Lock / Redis atomics.
    """

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    def create(self, job_type: JobType | None = None) -> Job:
        job = Job(type=job_type)
        self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def all(self) -> list[Job]:
        return list(self._jobs.values())

    def delete(self, job_id: str) -> bool:
        return self._jobs.pop(job_id, None) is not None


# Module-level singleton — imported by routes
job_store = JobStore()
