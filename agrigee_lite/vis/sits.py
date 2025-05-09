import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import Polygon

from agrigee_lite.get.sits import download_multiple_sits_chunks_multithread, download_single_sits
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
    axis: plt.Axes | None = None,
    color: str = "blue",
    alpha: float = 1,
) -> None:
    sits = download_single_sits(geometry, start_date, end_date, satellite, date_types=[date_type])
    long_sits = wide_to_long_dataframe(sits)
    band_columns = long_sits.columns[long_sits.columns != date_type]
    long_sits[band_columns] = satellite.scaleBands(long_sits[band_columns])

    if band_or_indice_to_plot in ALL_NUMPY_INDICES:
        y = compute_index_from_df(long_sits, ALL_NUMPY_INDICES[band_or_indice_to_plot])
    else:
        y = long_sits[band_or_indice_to_plot].values

    if axis is None:
        plt.plot(
            long_sits[date_type],
            y,
            color=color,
            alpha=alpha,
        )
        plt.scatter(
            long_sits[date_type],
            y,
            color=color,
        )
    else:
        axis.plot(long_sits[date_type], y, color=color, alpha=alpha, label=satellite.shortName)
        axis.scatter(
            long_sits[date_type],
            y,
            color=color,
        )


def visualize_multiple_sits(
    gdf: gpd.GeoDataFrame,
    band_or_indice_to_plot: str,
    satellite: AbstractSatellite,
    axis: plt.Axes | None = None,
    color: str = "blue",
    alpha: float = 0.5,
) -> None:
    sits = download_multiple_sits_chunks_multithread(gdf, satellite, date_types=["fyear"])
    long_sits = wide_to_long_dataframe(sits)

    if band_or_indice_to_plot in ALL_NUMPY_INDICES:
        long_sits["y"] = compute_index_from_df(long_sits, ALL_NUMPY_INDICES[band_or_indice_to_plot])

    long_sits = long_sits[long_sits.fyear != 0].reset_index(drop=True)

    for indexnumm in long_sits.indexnum.unique():
        indexnumm_df = long_sits[long_sits.indexnum == indexnumm].reset_index(drop=True)
        indexnumm_df["fyear"] = round(indexnumm_df.fyear.max()) - indexnumm_df.fyear
        y = (
            indexnumm_df["y"]
            if band_or_indice_to_plot in ALL_NUMPY_INDICES
            else indexnumm_df[band_or_indice_to_plot].values
        )
        if axis is None:
            plt.plot(
                indexnumm_df["fyear"],
                y,
                color=color,
                alpha=alpha,
            )
        else:
            axis.plot(indexnumm_df["fyear"], y, color=color, alpha=alpha, label=satellite.shortName)
