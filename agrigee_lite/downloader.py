import pathlib
import subprocess
import time
from importlib.resources import files

import aria2p


class DownloaderStrategy:
    """Thin wrapper around an aria2 RPC server that manages parallel downloads.

    On construction, starts a local ``aria2c`` process (if one is not already
    running) and connects to it via the aria2 JSON-RPC API on port 6800.  All
    download requests are forwarded to aria2, which handles connection pooling,
    retries, and resume automatically.

    Downloads are tracked by a caller-supplied ``my_id`` (int or str) rather
    than by GID, so the caller can correlate download status back to its own
    data model (e.g., chunk indices).

    Parameters
    ----------
    download_folder : pathlib.Path
        Directory where downloaded files will be saved.

    Notes
    -----
    The aria2 process is terminated when the object is garbage-collected.
    Only one aria2 process is expected per Python process; calling
    ``DownloaderStrategy`` a second time while aria2 is already running will
    reuse the existing process.
    """

    def __init__(self, download_folder: pathlib.Path):
        self.aria2 = aria2p.API(aria2p.Client(host="http://localhost", port=6800, secret=""))
        self.download_folder = download_folder
        conf_path = files("agrigee_lite").joinpath("aria2.conf")
        self.aria2_process = None

        if not self.is_downloader_running():
            self.aria2_process = subprocess.Popen(  # noqa: S603
                ["aria2c", f"--conf-path={conf_path}"],  # noqa: S607
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            while not self.is_downloader_running():
                time.sleep(1)

        self.downloads_map: dict[int | str, str] = {}  # {my_id: gid}
        self.retry_count: dict[int | str, int] = {}  # {my_id: num_retries}

    def __del__(self) -> None:
        if self.aria2_process is not None:
            try:
                self.aria2_process.terminate()
                self.aria2_process.wait(timeout=5)
            except:  # noqa: E722
                try:  # noqa: SIM105
                    self.aria2_process.kill()
                except:  # noqa: E722, S110
                    pass

    def is_downloader_running(self) -> bool:
        try:
            self.aria2.get_downloads()
        except:  # noqa: E722
            return False
        return True

    def add_download(self, items: list[tuple[int | str, str]]) -> None:
        for my_id, url in items:
            if my_id in self.downloads_map:
                try:
                    existing_download = self.aria2.get_download(self.downloads_map[my_id])
                    if existing_download.status != "error":
                        raise Exception(  # noqa: TRY002, TRY003, TRY301
                            f"Download with id={my_id} already exists with status='{existing_download.status}'."
                        )
                except Exception as e:
                    raise Exception(f"Error checking existing download for id={my_id}: {e}")  # noqa: B904, TRY002, TRY003

            download = self.aria2.add_uris([url], {"dir": str(self.download_folder.absolute()) + "/"})
            self.downloads_map[my_id] = download.gid
            if my_id not in self.retry_count:
                self.retry_count[my_id] = 0

    def _get_downloads_snapshot(self) -> dict[int | str, aria2p.Download]:
        """Fetch all tracked downloads in a single aria2 RPC call.

        All status-query methods on this class use this snapshot so that a
        single loop iteration never issues more than one RPC call regardless
        of how many chunks are tracked.
        """
        if not self.downloads_map:
            return {}
        gid_to_id = {gid: my_id for my_id, gid in self.downloads_map.items()}
        return {gid_to_id[d.gid]: d for d in self.aria2.get_downloads() if d.gid in gid_to_id}

    @property
    def downloads(self) -> dict[int | str, aria2p.Download]:
        return self._get_downloads_snapshot()

    @property
    def num_unfinished_downloads(self) -> int:
        return sum(d.status not in ("complete", "error") for d in self._get_downloads_snapshot().values())

    @property
    def still_downloading_ids(self) -> list[str]:
        return [str(my_id) for my_id, d in self._get_downloads_snapshot().items() if d.status not in ("complete", "error")]

    @property
    def num_downloads_with_error(self) -> int:
        return sum(d.status == "error" for d in self._get_downloads_snapshot().values())

    @property
    def num_completed_downloads(self) -> int:
        return sum(d.status == "complete" for d in self._get_downloads_snapshot().values())

    @property
    def failed_downloads(self) -> list[int | str]:
        return [my_id for my_id, d in self._get_downloads_snapshot().items() if d.status == "error"]

    def get_failed_downloads_within_retry_limit(self, max_retries: int) -> list[int | str]:
        return [
            my_id
            for my_id, d in self._get_downloads_snapshot().items()
            if d.status == "error" and self.retry_count.get(my_id, 0) < max_retries
        ]

    def get_failed_downloads_exceeding_retry_limit(self, max_retries: int) -> list[int | str]:
        return [
            my_id
            for my_id, d in self._get_downloads_snapshot().items()
            if d.status == "error" and self.retry_count.get(my_id, 0) >= max_retries
        ]

    def get_num_failed_downloads_exceeding_retry_limit(self, max_retries: int) -> int:
        return len(self.get_failed_downloads_exceeding_retry_limit(max_retries))

    def increment_retry_count(self, my_id: int | str) -> None:
        self.retry_count[my_id] = self.retry_count.get(my_id, 0) + 1

    @property
    def is_empty(self) -> bool:
        return len(self.downloads_map) == 0

    def reset_downloads(self) -> None:
        self.downloads_map.clear()
        self.retry_count.clear()

    def stats_from_snapshot(self, snapshot: dict[int | str, aria2p.Download]) -> dict[str, int]:
        """Return common counters from a pre-fetched snapshot.

        Callers that need multiple counters in the same loop iteration should
        call ``_get_downloads_snapshot()`` once, then pass the result here to
        avoid redundant RPC calls.
        """
        return {
            "completed": sum(d.status == "complete" for d in snapshot.values()),
            "errors": sum(d.status == "error" for d in snapshot.values()),
            "active": sum(d.status not in ("complete", "error") for d in snapshot.values()),
        }
