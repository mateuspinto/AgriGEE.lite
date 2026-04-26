import logging

from agrigee_lite.cache import clear_cache, init_cache, print_cache_status
from agrigee_lite.ee_utils import ee_get_tasks_status as get_all_tasks
from agrigee_lite.ee_utils import ee_quick_start
from agrigee_lite.misc import get_sample_gdf, h3_clustering, random_points_from_gdf

from . import (
    get,
    sat,
    vis,
)

__all__ = [
    "clear_cache",
    "ee_quick_start",
    "get",
    "get_all_tasks",
    "get_sample_gdf",
    "initialize",
    "h3_clustering",
    "random_points_from_gdf",
    "sat",
    "vis",
]


def initialize():
    init_cache()
    print_cache_status()
    ee_quick_start()
