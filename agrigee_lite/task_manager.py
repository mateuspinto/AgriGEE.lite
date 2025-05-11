import time
from datetime import datetime

import ee
import pandas as pd
from tqdm.std import tqdm


class GEETaskManager:
    def __init__(self) -> None:
        self.unstarted_tasks: list[ee.batch.Task] = []
        self.started_tasks: list[ee.batch.Task] = []
        self.other_tasks = pd.DataFrame()
        self.last_checked = datetime(1999, 12, 4)

    def add(self, task: ee.batch.Task) -> None:
        self.unstarted_tasks.append(task)

    def start(self) -> None:
        for task in self.unstarted_tasks:
            task.start()
            self.started_tasks.append(task)
        self.unstarted_tasks = []

    def wait(self) -> None:
        failed_count = 0
        canceled_count = 0

        with tqdm(total=len(self.started_tasks), desc="Waiting for tasks") as pbar:
            while self.started_tasks:
                for task in self.started_tasks:
                    task_status = task.status()
                    if task_status["state"] == "COMPLETED":
                        self.started_tasks.remove(task)
                        pbar.update(1)
                    elif task_status["state"] == "FAILED":
                        self.started_tasks.remove(task)
                        pbar.update(1)
                        failed_count += 1
                        pbar.set_postfix_str(f"Failed tasks: {failed_count}")
                    elif task_status["state"] in {"CANCELLING", "CANCELED"}:
                        self.started_tasks.remove(task)
                        pbar.update(1)
                        canceled_count += 1
                        pbar.set_postfix_str(f"Canceled tasks: {canceled_count}")
                    else:
                        pass

                time.sleep(10)
