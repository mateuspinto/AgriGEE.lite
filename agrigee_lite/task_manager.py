from datetime import datetime

import ee
import pandas as pd


class GEETaskManager:
    def __init__(self) -> None:
        self.unstarted_tasks: list[ee.batch.Task] = []
        self.other_tasks = pd.DataFrame()
        self.last_checked = datetime(1999, 12, 4)

    def add(self, task: ee.batch.Task) -> None:
        self.unstarted_tasks.append(task)
