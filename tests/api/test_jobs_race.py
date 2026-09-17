"""Regression test for the DuckDB duplicate-key crash in JobStore.create().

Reproduces the check-then-insert race that submit_multiple_sits_job /
submit_images_job leave open around JobStore.create(): two submissions
computing the same content-hash job id both see `job_store.get(id) is None`
and both call `create(job_id=id)`. Before this fix, the second INSERT hit
api_jobs' PRIMARY KEY constraint and DuckDB escalated that into a
`duckdb.duckdb.FatalException`, aborting the whole process. See
create_api_job's docstring in agrigee_lite/cache/backend.py.
"""

from __future__ import annotations

import pathlib

from agrigee_lite.api._jobs import Job, JobStatus, JobStore, JobType
from agrigee_lite.cache import init_cache


def _new_store(db_path: pathlib.Path) -> JobStore:
    init_cache(db_path)
    store = JobStore()
    store.load_from_db()  # mirrors _lifespan(): migrates the api_jobs schema
    return store


def test_create_with_duplicate_id_does_not_raise(tmp_path: pathlib.Path) -> None:
    store = _new_store(tmp_path / "jobs.duckdb")

    first = store.create(JobType.SITS, job_id="same-hash")
    # Simulates a racing second submission for the exact same content hash —
    # this used to reach a raw INSERT and crash the process.
    second = store.create(JobType.SITS, job_id="same-hash")

    assert first.id == second.id == "same-hash"
    assert len(store.all()) == 1


def test_duplicate_create_after_first_completes_reuses_existing_row(tmp_path: pathlib.Path) -> None:
    store = _new_store(tmp_path / "jobs.duckdb")

    job = store.create(JobType.IMAGES, job_id="same-hash")
    fake_cache_dir = str(tmp_path / "images_cache")
    store.update_status(job.id, JobStatus.COMPLETED, result={"cache_dir": fake_cache_dir, "dates": ["2026-01-01"]})

    again = store.create(JobType.IMAGES, job_id="same-hash")

    assert again.status == JobStatus.COMPLETED
    assert again.result == {"cache_dir": fake_cache_dir, "dates": ["2026-01-01"]}


def test_create_wins_the_insert_when_id_is_new(tmp_path: pathlib.Path) -> None:
    store = _new_store(tmp_path / "jobs.duckdb")

    job = store.create(JobType.SITS, job_id="brand-new")

    assert job.id == "brand-new"
    assert job.status == JobStatus.PENDING
    assert store.get("brand-new") is job


def test_create_recovers_when_db_row_exists_but_memory_does_not(tmp_path: pathlib.Path) -> None:
    """The process-restart edge case: create_api_job() reports a conflict for
    an id this fresh JobStore instance has never loaded into memory (e.g. a
    row from before load_from_db() ran). create() must hydrate from the DB
    instead of raising a KeyError.

    Uses COMPLETED rather than RUNNING so the assertion isn't entangled with
    load_from_db()'s separate, correct behavior of resetting orphaned RUNNING
    jobs to FAILED on load (see JobStore.load_from_db's docstring).
    """
    db_path = tmp_path / "jobs.duckdb"
    engine = init_cache(db_path)
    JobStore().load_from_db()  # runs the api_jobs schema migration once

    from agrigee_lite.cache.backend import create_api_job

    create_api_job(engine, "orphan-hash", JobType.SITS.value, JobStatus.COMPLETED.value, "2026-01-01T00:00:00+00:00")

    store = JobStore()  # deliberately not load_from_db()'d
    job = store.create(JobType.SITS, job_id="orphan-hash")

    assert isinstance(job, Job)
    assert job.id == "orphan-hash"
    assert job.status == JobStatus.COMPLETED
