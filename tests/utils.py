import numpy as np

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


def check_np_array_equivalence(arr1: np.ndarray, arr2: np.ndarray, threshold: float = 0.1) -> bool:
    return bool(np.all((arr1 >= arr2 * (1 - threshold)) & (arr1 <= arr2 * (1 + threshold))))


def get_all_satellites_for_test() -> list[AbstractSatellite]:
    return [agl.sat.Sentinel2(), agl.sat.Sentinel2(use_sr=True)]
