import pathlib
import subprocess
import time
from importlib.resources import files

import aria2p


class DownloaderStrategy:
    def __init__(self, download_folder: pathlib.Path):
        self.aria2 = aria2p.API(aria2p.Client(host="http://localhost", port=6800, secret=""))
        self.download_folder = download_folder
        conf_path = files("agrigee_lite").joinpath("aria2.conf")

        if not self.is_downloader_running():
            subprocess.Popen(  # noqa: S603
                ["aria2c", f"--conf-path={conf_path}"],  # noqa: S607
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            while not self.is_downloader_running():
                time.sleep(1)

        self.downloads_map = {}  # {my_id: gid}
        self.retry_count = {}  # {my_id: num_retries}

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

    @property
    def downloads(self) -> dict[str, aria2p.Download]:
        items_copy = list(self.downloads_map.items())
        return {my_id: self.aria2.get_download(gid) for my_id, gid in items_copy}

    @property
    def num_unfinished_downloads(self) -> int:
        return sum(d.status == "active" for d in self.downloads.values())

    @property
    def num_downloads_with_error(self) -> int:
        return sum(d.status == "error" for d in self.downloads.values())

    @property
    def num_completed_downloads(self) -> int:
        return sum(d.status == "complete" for d in self.downloads.values())

    @property
    def failed_downloads(self) -> list[str]:
        return [my_id for my_id, d in self.downloads.items() if d.status == "error"]

    def get_failed_downloads_within_retry_limit(self, max_retries: int) -> list[str]:
        return [
            my_id
            for my_id, d in self.downloads.items()
            if d.status == "error" and self.retry_count.get(my_id, 0) < max_retries
        ]

    def increment_retry_count(self, my_id: int | str) -> None:
        self.retry_count[my_id] = self.retry_count.get(my_id, 0) + 1

    @property
    def is_empty(self) -> bool:
        return len(self.downloads_map) == 0

    def reset_downloads(self) -> None:
        self.downloads_map.clear()
        self.retry_count.clear()
