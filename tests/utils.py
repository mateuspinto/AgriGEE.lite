import geopandas as gpd
import numpy as np
import pandas as pd

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


def assert_np_array_equivalence(arr1: np.ndarray, arr2: np.ndarray, threshold: float = 0.1) -> None:
    arr1 = arr1.flatten()
    arr2 = arr2.flatten()

    if arr1.shape != arr2.shape:
        raise AssertionError(f"Shape mismatch: {arr1.shape} vs {arr2.shape}")

    bound1 = arr2 * (1 - threshold)
    bound2 = arr2 * (1 + threshold)
    lower_bound = np.minimum(bound1, bound2)
    upper_bound = np.maximum(bound1, bound2)

    valid = (arr1 >= lower_bound) & (arr1 <= upper_bound)

    if not np.all(valid):
        invalid_indices = np.where(~valid)[0]
        error_pct = 100 * len(invalid_indices) / len(arr1)

        msg_lines = [
            f"Arrays differ: {len(invalid_indices)} of {len(arr1)} values outside threshold ({error_pct:.2f}%)",
        ]

        for i in invalid_indices[:5]:
            msg_lines.append(
                f"[{i}] Downloaded={arr1[i]:.4f}, Original={arr2[i]:.4f}, "
                f"Allowed=[{lower_bound[i]:.4f}, {upper_bound[i]:.4f}]"
            )

        raise AssertionError("\n".join(msg_lines))


def assert_df_equivalence(
    df1: pd.DataFrame | gpd.GeoDataFrame, df2: pd.DataFrame | gpd.GeoDataFrame, threshold: float = 0.1
) -> None:
    cols1 = set(df1.columns)
    cols2 = set(df2.columns)

    missing_in_df2 = cols1 - cols2
    extra_in_df2 = cols2 - cols1

    if missing_in_df2 or extra_in_df2:
        msg = ["Column mismatch detected:"]
        if missing_in_df2:
            msg.append(f"- Missing in second DataFrame: {sorted(missing_in_df2)}")
        if extra_in_df2:
            msg.append(f"- Additional in second DataFrame: {sorted(extra_in_df2)}")
        raise AssertionError("\n".join(msg))

    arr1 = df1[sorted(cols1)].to_numpy()
    arr2 = df2[sorted(cols1)].to_numpy()

    assert_np_array_equivalence(arr1, arr2, threshold)


def get_all_satellites_for_test() -> list[AbstractSatellite]:
    return [
        agl.sat.Sentinel2(),
        agl.sat.Sentinel2(use_sr=True),
        agl.sat.Sentinel1(),
        agl.sat.Sentinel1(ascending=False),
    ]


def get_all_date_types_for_test() -> list[str]:
    return ["doy", "year", "fyear"]


def get_all_reducers_for_test() -> list[str]:
    return ["min", "max", "mean", "median", "mode", "std", "var", "p2", "p98", "kurt", "skew"]
