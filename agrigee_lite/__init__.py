from warnings import simplefilter

import pandas as pd

from agrigee_lite import get, sat, vis
from agrigee_lite.ee_utils import ee_get_tasks_status as get_all_tasks
from agrigee_lite.misc import quadtree_clustering

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.copy_on_write = True
