import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely import Polygon

from agrigee_lite.get.sits import download_multiple_sits_chunks_multithread, download_single_sits
from agrigee_lite.misc import compute_index_from_df
from agrigee_lite.numpy_indices import ALL_NUMPY_INDICES
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


def year_fraction(dt: pd.Series) -> pd.Series:
    year = dt.year
    start_of_year = pd.Timestamp(year=year, month=1, day=1)
    end_of_year = pd.Timestamp(year=year + 1, month=1, day=1)
    fraction = (dt - start_of_year) / (end_of_year - start_of_year)
    return year + fraction


def visualize_single_sits(
    geometry: Polygon,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    band_or_indice_to_plot: str,
    reducer: str = "median",
    ax: plt.Axes | None = None,
    color: str = "blue",
    alpha: float = 1,
) -> None:
    long_sits = download_single_sits(geometry, start_date, end_date, satellite, reducers=[reducer])

    if len(long_sits) == 0:
        return None

    if band_or_indice_to_plot in ALL_NUMPY_INDICES:
        y = compute_index_from_df(long_sits, ALL_NUMPY_INDICES[band_or_indice_to_plot])
    else:
        y = long_sits[band_or_indice_to_plot].values

    long_sits["timestamp"] = pd.to_datetime(long_sits["timestamp"])

    if ax is None:
        plt.plot(
            long_sits.timestamp,
            y,
            color=color,
            alpha=alpha,
        )
        plt.scatter(
            long_sits.timestamp,
            y,
            color=color,
        )
    else:
        ax.plot(long_sits.timestamp, y, color=color, alpha=alpha, label=satellite.shortName)
        ax.scatter(
            long_sits.timestamp,
            y,
            color=color,
        )


def visualize_multiple_sits(
    gdf: gpd.GeoDataFrame,
    satellite: AbstractSatellite,
    band_or_indice_to_plot: str,
    reducer: str = "median",
    ax: plt.Axes | None = None,
    color: str = "blue",
    alpha: float = 0.5,
) -> None:
    long_sits = download_multiple_sits_chunks_multithread(gdf, satellite, reducers=[reducer])

    if len(long_sits) == 0:
        return None

    if band_or_indice_to_plot in ALL_NUMPY_INDICES:
        long_sits["y"] = compute_index_from_df(long_sits, ALL_NUMPY_INDICES[band_or_indice_to_plot])

    for indexnumm in long_sits.indexnum.unique():
        indexnumm_df = long_sits[long_sits.indexnum == indexnumm].reset_index(drop=True).copy()
        indexnumm_df["timestamp"] = indexnumm_df.timestamp.apply(year_fraction)
        indexnumm_df["timestamp"] = indexnumm_df["timestamp"] - indexnumm_df["timestamp"].min().round()

        y = (
            indexnumm_df["y"]
            if band_or_indice_to_plot in ALL_NUMPY_INDICES
            else indexnumm_df[band_or_indice_to_plot].values
        )

        if ax is None:
            plt.plot(
                indexnumm_df.timestamp,
                y,
                color=color,
                alpha=alpha,
            )
        else:
            ax.plot(indexnumm_df.timestamp, y, color=color, alpha=alpha, label=satellite.shortName)
