import pandas as pd
from shapely import Polygon

from agrigee_lite.sat.abstract_satellite import OpticalSatellite


def visualize_multiple_images(
    geometry: Polygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: OpticalSatellite,
    invalid_images_threshold: float = 0.5,
    contrast: float = 1.3,
    num_threads_rush: int = 30,
    num_threads_retry: int = 10,
) -> None:
    raise NotImplementedError("This function is not implemented yet.")
