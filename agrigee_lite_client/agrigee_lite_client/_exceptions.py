"""Exception hierarchy for agrigee_lite_client. See SPECS.md §7."""

from __future__ import annotations


class AgriGEEClientError(Exception):
    """Base class for every error this client raises."""


class AgriGEEHTTPError(AgriGEEClientError):
    """The server returned an HTTP error status."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


class AgriGEEJobError(AgriGEEClientError):
    """A submitted job finished with status='failed'."""

    def __init__(self, job_id: str, error: str | None) -> None:
        self.job_id = job_id
        self.error = error
        super().__init__(f"Job {job_id} failed: {error}")


class AgriGEEJobTimeoutError(AgriGEEClientError):
    """poll_timeout elapsed before the job reached a terminal status."""

    def __init__(self, job_id: str, poll_timeout: float) -> None:
        self.job_id = job_id
        self.poll_timeout = poll_timeout
        super().__init__(f"Job {job_id} did not finish within {poll_timeout}s")


class AgriGEEVersionMismatchError(AgriGEEClientError):
    """The server's agrigee_lite version doesn't match this client's.

    Raised before trusting the client's embedded satellite date table
    (SPECS.md §5.5) — that table is a static copy of server-side data, only
    guaranteed correct when both sides are on the same version.
    """

    def __init__(self, client_version: str, server_version: str | None) -> None:
        self.client_version = client_version
        self.server_version = server_version
        reported = server_version if server_version is not None else "an unknown version (no 'version' in /version)"
        super().__init__(
            f"agrigee_lite_client is version {client_version} but the server is running {reported}. "
            "Install matching versions of both — the client's satellite date table is only valid "
            "when they agree."
        )


class AgriGEEUnknownSatelliteError(AgriGEEClientError):
    """A satellite name isn't in this client's embedded date-range table."""
