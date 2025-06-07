import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
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


def adjust_doys(doys: np.ndarray) -> np.ndarray:
    """
    Adjusts a sequence of DOYs by detecting the first drop (a value smaller than the one before)
    and adding 365 to all values from that point onward.

    Parameters
    ----------
    doys : np.ndarray
        1D array of DOY (day-of-year) values as integers.

    Returns
    -------
    np.ndarray
        A new array with the adjusted DOY values.
    """
    doys = doys.copy()
    diffs = np.diff(doys)
    break_idx = np.where(diffs < 0)[0]

    if break_idx.size > 0:
        first_break = break_idx[0] + 1
        doys[first_break:] += 365

    return doys


def visualize_multiple_sits(
    gdf: gpd.GeoDataFrame,
    band_or_indice_to_plot: str,
    satellite: AbstractSatellite,
    axis: plt.Axes | None = None,
    color: str = "blue",
    alpha: float = 0.5,
) -> None:
    if f"{satellite.shortName}_observations" in gdf.columns.tolist():
        print("Columns already existing, not downloading again")
        sits = gdf[sorted(filter(lambda x: x.startswith(satellite.shortName), gdf.columns.tolist()))]
    else:
        sits = download_multiple_sits_chunks_multithread(gdf, satellite, date_types=["fyear"])

    long_sits = wide_to_long_dataframe(sits)

    if band_or_indice_to_plot in ALL_NUMPY_INDICES:
        long_sits["y"] = compute_index_from_df(long_sits, ALL_NUMPY_INDICES[band_or_indice_to_plot])

    long_sits = long_sits[long_sits.doy != 0].reset_index(drop=True)

    for indexnumm in long_sits.indexnum.unique():
        indexnumm_df = long_sits[long_sits.indexnum == indexnumm].reset_index(drop=True)
        indexnumm_df = indexnumm_df.dropna(
            subset=["doy", "green", "nir", "re1", "re2", "re3", "re4", "red", "swir1"]
        ).reset_index(drop=True)

        indexnumm_df["doy"] = indexnumm_df["doy"].astype(int)

        doys = indexnumm_df["doy"].to_numpy()
        doys = adjust_doys(doys)

        y = (
            indexnumm_df["y"]
            if band_or_indice_to_plot in ALL_NUMPY_INDICES
            else indexnumm_df[band_or_indice_to_plot].values
        )

        if axis is None:
            plt.plot(
                doys,
                y,
                color=color,
                alpha=alpha,
            )
        else:
            axis.plot(doys, y, color=color, alpha=alpha, label=satellite.shortName)
