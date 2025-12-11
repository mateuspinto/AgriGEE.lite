from typing import Any

import geopandas as gpd
import pandas as pd
from shapely import MultiPolygon, Point, Polygon

from agrigee_lite.get.sits import download_multiple_sits, download_single_sits
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


def year_fraction(dt: pd.Series) -> pd.Series:
    """
    Convert datetime series to year fraction format for temporal visualization.

    This function converts timestamps to a continuous year representation where
    the fractional part represents the position within the year (0.0 = Jan 1st,
    0.5 ≈ July 1st, etc.). This is useful for overlaying multiple years of data
    in time series visualizations.

    Parameters
    ----------
    dt : pd.Series
        Series of datetime objects to convert.

    Returns
    -------
    pd.Series
        Series of float values representing year fractions (e.g., 2022.5 for mid-2022).
    """
    year = dt.year
    start_of_year = pd.Timestamp(year=year, month=1, day=1)
    end_of_year = pd.Timestamp(year=year + 1, month=1, day=1)
    fraction = (dt - start_of_year) / (end_of_year - start_of_year)
    return year + fraction


def visualize_single_sits(
    geometry: Polygon | MultiPolygon | Point,
    start_date: pd.Timestamp | str,
    end_date: pd.Timestamp | str,
    satellite: AbstractSatellite,
    band_or_indice_to_plot: str,
    reducer: str = "median",
    ax: Any = None,
    color: str = "blue",
    alpha: float = 1,
) -> None:
    """
    Visualize satellite time series for a single geometry.

    Creates a line plot with scatter points showing the temporal evolution of a
    specified band or vegetation index for a single geometric area. The function
    downloads the satellite time series data and plots it using matplotlib.

    Parameters
    ----------
    geometry : Polygon, MultiPolygon, or Point
        The area or point of interest for time series extraction.
    start_date : pd.Timestamp or str
        Start date for the time series.
    end_date : pd.Timestamp or str
        End date for the time series.
    satellite : AbstractSatellite
        Satellite configuration object.
    band_or_indice_to_plot : str
        Name of the band or vegetation index to visualize.
    reducer : str, optional
        Temporal reducer to apply (e.g., "median", "mean"), by default "median".
    ax : matplotlib.axes.Axes or None, optional
        Matplotlib axes object for plotting. If None, creates a new plot, by default None.
    color : str, optional
        Color for the plot line and points, by default "blue".
    alpha : float, optional
        Transparency level (0.0 to 1.0), by default 1.

    Returns
    -------
    None
        The function creates a plot but doesn't return any value.
    """
    import matplotlib.pyplot as plt

    long_sits = download_single_sits(geometry, start_date, end_date, satellite, reducers={reducer})

    if len(long_sits) == 0:
        return None

    y = long_sits[band_or_indice_to_plot].values

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
    ax: Any = None,
    color: str = "blue",
    alpha: float = 0.5,
    force_redownload: bool = False,
) -> None:
    """
    Visualize satellite time series for multiple geometries with normalized temporal alignment.

    Creates overlaid line plots for multiple geometries, with time series normalized
    to year fractions to enable comparison across different years. Each geometry's
    time series is plotted as a semi-transparent line, making it easy to identify
    patterns and outliers across the dataset.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame containing multiple geometries and their temporal information.
        Must have the required date columns for satellite time series processing.
    satellite : AbstractSatellite
        Satellite configuration object.
    band_or_indice_to_plot : str
        Name of the band or vegetation index to visualize.
    reducer : str, optional
        Temporal reducer to apply (e.g., "median", "mean"), by default "median".
    ax : matplotlib.axes.Axes or None, optional
        Matplotlib axes object for plotting. If None, creates a new plot, by default None.
    color : str, optional
        Color for the plot lines, by default "blue".
    alpha : float, optional
        Transparency level for individual lines (0.0 to 1.0), by default 0.5.
        Lower values help visualize overlapping time series.

    Returns
    -------
    None
        The function creates a plot but doesn't return any value.

    Notes
    -----
    This function normalizes timestamps to year fractions, where each time series
    starts from 0.0, making it possible to overlay multiple years of data for
    pattern analysis. The original timestamps are converted using the `year_fraction`
    function and then normalized to start from zero.
    """
    import matplotlib.pyplot as plt

    long_sits = download_multiple_sits(gdf, satellite, reducers={reducer}, force_redownload=force_redownload)

    if len(long_sits) == 0:
        return None

    for indexnumm in long_sits.original_index.unique():
        indexnumm_df = long_sits[long_sits.original_index == indexnumm].reset_index(drop=True).copy()
        indexnumm_df["timestamp"] = indexnumm_df.timestamp.apply(year_fraction)
        indexnumm_df["timestamp"] = indexnumm_df["timestamp"] - indexnumm_df["timestamp"].min().round()

        y = indexnumm_df[band_or_indice_to_plot].values

        if ax is None:
            plt.plot(
                indexnumm_df.timestamp,
                y,
                color=color,
                alpha=alpha,
            )
        else:
            ax.plot(indexnumm_df.timestamp, y, color=color, alpha=alpha, label=satellite.shortName)
