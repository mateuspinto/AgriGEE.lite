import pathlib
import subprocess
import time

import aria2p


class DownloaderStrategy:
    def __init__(self, download_folder: pathlib.Path):
        self.aria2 = aria2p.API(aria2p.Client(host="http://localhost", port=6800, secret=""))
        self.download_folder = download_folder

        if not self.is_downloader_running():
            subprocess.Popen(  # noqa: S603
                ["aria2c", "--conf-path=aria2.conf"],  # noqa: S607
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            while not self.is_downloader_running():
                time.sleep(1)

        self.download_gids = []

    def is_downloader_running(self) -> bool:
        try:
            self.aria2.get_downloads()
        except:  # noqa: E722
            return False

        return True

    def add_download(self, urls: list[str]) -> None:
        self.download_gids.append(self.aria2.add_uris(urls, {"dir": str(self.download_folder.absolute()) + "/"}).gid)

    @property
    def downloads(self) -> list[aria2p.Download]:
        return [self.aria2.get_download(gid) for gid in self.download_gids]

    @property
    def num_unfinished_downloads(self) -> int:
        return sum(d.status == "active" for d in self.downloads)

    @property
    def num_downloads_with_error(self) -> int:
        return sum(d.status == "error" for d in self.downloads)

    @property
    def num_completed_downloads(self) -> int:
        return sum(d.status == "complete" for d in self.downloads)

    @property
    def failed_downloads(self) -> list[int]:
        return [i for i, d in enumerate(self.downloads) if d.status == "error"]

    @property
    def is_empty(self) -> bool:
        return len(self.downloads) == 0

    def reset_downloads(self) -> None:
        self.download_gids = []
