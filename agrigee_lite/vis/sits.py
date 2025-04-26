import matplotlib.pyplot as plt
import pandas as pd
from shapely import Polygon

from agrigee_lite.get.sits import download_single_sits
from agrigee_lite.misc import compute_index_from_df, wide_to_long_dataframe
from agrigee_lite.numpy_indices import ALL_NUMPY_INDICES
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


def visualize_single_sits(
    geometry: Polygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    band_or_indice_to_plot: str,
    date_type: str = "fyear",
) -> None:
    sits = download_single_sits(geometry, start_date, end_date, satellite, date_types=[date_type])
    long_sits = wide_to_long_dataframe(sits)
    band_columns = long_sits.columns[long_sits.columns != date_type]
    long_sits[band_columns] = satellite.scaleBands(long_sits[band_columns])

    if band_or_indice_to_plot in ALL_NUMPY_INDICES:
        y = compute_index_from_df(long_sits, ALL_NUMPY_INDICES[band_or_indice_to_plot])
    else:
        y = long_sits[band_or_indice_to_plot].values

    plt.plot(long_sits[date_type], y)
