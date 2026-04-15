import time
from datetime import datetime

import ee
import pandas as pd
from tqdm.std import tqdm


class GEETaskManager:
    """Manage and monitor GEE batch export tasks.

    GEE can run long-running export jobs (images to Drive, Cloud Storage, etc.)
    as server-side *tasks*.  This class queues tasks, starts them, and polls
    their status until all are completed, printing a progress bar.

    Typical usage::

        manager = GEETaskManager()
        manager.add(some_task)
        manager.start()
        manager.wait()  # blocks until every task is COMPLETED/FAILED/CANCELED

    Notes
    -----
    This class is used internally by the GDrive / GCS download paths.
    Direct use is only needed when you want fine-grained control over task
    execution outside of the standard download functions.
    """

    def __init__(self) -> None:
        self.unstarted_tasks: list[ee.batch.Task] = []
        self.started_tasks: list[ee.batch.Task] = []
        self.other_tasks = pd.DataFrame()
        self.last_checked = datetime(1999, 12, 4)

    def add(self, task: ee.batch.Task) -> None:
        """Queue a GEE task for later execution.

        Parameters
        ----------
        task : ee.batch.Task
            A GEE export task created via ``ee.batch.Export.*``.
            The task is not started until :meth:`start` is called.
        """
        self.unstarted_tasks.append(task)

    def start(self) -> None:
        """Submit all queued tasks to GEE and move them to the started list."""
        for task in self.unstarted_tasks:
            task.start()
            self.started_tasks.append(task)
        self.unstarted_tasks = []

    def wait(self) -> None:
        """Block until every started task reaches a terminal state.

        Polls task status every 10 seconds and updates a tqdm progress bar.
        Failed and cancelled tasks are counted and shown in the progress bar
        postfix but do not raise an exception.
        """
        failed_count = 0
        canceled_count = 0

        with tqdm(total=len(self.started_tasks), desc="Waiting for tasks") as pbar:
            while self.started_tasks:
                still_running: list[ee.batch.Task] = []
                for task in self.started_tasks:
                    task_status = task.status()
                    if task_status["state"] == "COMPLETED":
                        pbar.update(1)
                    elif task_status["state"] == "FAILED":
                        pbar.update(1)
                        failed_count += 1
                        pbar.set_postfix_str(f"Failed tasks: {failed_count}")
                    elif task_status["state"] in {"CANCELLING", "CANCELED"}:
                        pbar.update(1)
                        canceled_count += 1
                        pbar.set_postfix_str(f"Canceled tasks: {canceled_count}")
                    else:
                        still_running.append(task)
                self.started_tasks = still_running

                time.sleep(10)
