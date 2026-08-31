"""Stateful httpx.MockTransport handler faking just enough of the real
agrigee_lite API (health, version, satellites, the sits/images job
lifecycle) to drive the real client code end-to-end without a live server.

The same handler works for both httpx.Client and httpx.AsyncClient — a
plain sync callable is valid for MockTransport either way.
"""

from __future__ import annotations

import email
import io
import json
import zipfile
from dataclasses import dataclass, field

import httpx
import polars as pl

from agrigee_lite_client._version import __version__


def _extract_multipart_file(request: httpx.Request, field_name: str) -> bytes:
    content_type = request.headers["content-type"]
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = email.message_from_bytes(header + request.read())
    for part in message.walk():
        if f'name="{field_name}"' in part.get("Content-Disposition", ""):
            payload = part.get_payload(decode=True)
            assert isinstance(payload, bytes)
            return payload
    msg = f"multipart field {field_name!r} not found in request"
    raise AssertionError(msg)


def _build_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


@dataclass
class _Job:
    kind: str  # "sits" | "images"
    remaining_statuses: list[str]
    error: str | None = None


@dataclass
class FakeServer:
    sits_result: pl.DataFrame = field(
        default_factory=lambda: pl.DataFrame({"timestamp": ["2023-01-01"], "B4": [0.42]})
    )
    image_zip: bytes = field(default_factory=lambda: _build_zip({"2023-01-01.tif": b"fake-tiff-bytes"}))
    sits_upload_rows: list[int] = field(default_factory=list)
    """Row count of every parquet uploaded to /sits/multiple/file, in call order."""
    image_requests: list[dict] = field(default_factory=list)
    """Every JSON body posted to /images, in call order."""
    server_version: str | None = __version__
    """Reported by /version — override to simulate a version mismatch/old server."""

    _jobs: dict[str, _Job] = field(default_factory=dict)
    _next_job_id: int = 0
    _next_sits_statuses: list[str] = field(default_factory=lambda: ["running", "completed"])
    _next_sits_error: str | None = None
    _next_image_statuses: list[str] = field(default_factory=lambda: ["running", "completed"])
    _next_image_error: str | None = None

    def queue_sits_job(self, statuses: list[str] | None = None, error: str | None = None) -> None:
        """Configure the status sequence the *next* sits submission will report."""
        self._next_sits_statuses = list(statuses) if statuses is not None else ["completed"]
        self._next_sits_error = error

    def queue_image_job(self, statuses: list[str] | None = None, error: str | None = None) -> None:
        self._next_image_statuses = list(statuses) if statuses is not None else ["completed"]
        self._next_image_error = error

    def _new_job(self, kind: str, statuses: list[str], error: str | None) -> str:
        self._next_job_id += 1
        job_id = f"job-{self._next_job_id}"
        self._jobs[job_id] = _Job(kind=kind, remaining_statuses=list(statuses), error=error)
        return job_id

    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path

        if method == "GET" and path == "/health":
            return httpx.Response(200, json={"status": "ok"})

        if method == "GET" and path == "/version":
            return httpx.Response(200, json={"version": self.server_version})

        if method == "GET" and path == "/satellites":
            return httpx.Response(200, json=["Landsat8", "Sentinel2"])

        if method == "POST" and path == "/sits/multiple/file":
            file_bytes = _extract_multipart_file(request, "file")
            self.sits_upload_rows.append(pl.read_parquet(io.BytesIO(file_bytes)).height)
            job_id = self._new_job("sits", self._next_sits_statuses, self._next_sits_error)
            return httpx.Response(202, json={"id": job_id, "type": "sits", "status": "pending"})

        if method == "POST" and path == "/images":
            self.image_requests.append(json.loads(request.read()))
            job_id = self._new_job("images", self._next_image_statuses, self._next_image_error)
            return httpx.Response(202, json={"id": job_id, "type": "images", "status": "pending"})

        if method == "GET" and path.startswith("/jobs/") and path.endswith("/download"):
            job_id = path.split("/")[2]
            job = self._jobs[job_id]
            if job.kind == "sits":
                buf = io.BytesIO()
                self.sits_result.write_parquet(buf)
                return httpx.Response(200, content=buf.getvalue())
            return httpx.Response(200, content=self.image_zip)

        if method == "GET" and path.startswith("/jobs/"):
            job_id = path.split("/")[2]
            job = self._jobs[job_id]
            status = job.remaining_statuses.pop(0) if len(job.remaining_statuses) > 1 else job.remaining_statuses[0]
            if status == "failed":
                return httpx.Response(200, json={"id": job_id, "type": job.kind, "status": "failed", "error": job.error})
            return httpx.Response(200, json={"id": job_id, "type": job.kind, "status": status})

        msg = f"unhandled request in FakeServer: {method} {path}"
        raise AssertionError(msg)
