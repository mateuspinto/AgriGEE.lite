import numpy as np
import pandas as pd

import agrigee_lite as agl
from agrigee_lite.sat.abstract_satellite import AbstractSatellite

# Maximum tolerable NaN fraction per satellite date type.
# Optical sensors are cloud-affected; static datasets should be nearly complete.
MAX_NAN_BY_DATE_TYPE: dict[str, float] = {
    "optical": 0.30,
    "radar": 0.20,
    "dataSource": 0.05,
    "singleImage": 0.05,
}


def assert_sits_quality(
    df: pd.DataFrame,
    satellite: AbstractSatellite,
    max_nan: float | None = None,
) -> None:
    """Assert that a sits DataFrame is non-empty and has an acceptable NaN fraction."""
    assert not df.empty, f"[{satellite.shortName}] Result is empty"

    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return

    nan_fraction = numeric.isna().to_numpy().mean()
    threshold = max_nan if max_nan is not None else MAX_NAN_BY_DATE_TYPE.get(satellite.dateType, 0.30)
    assert nan_fraction <= threshold, (
        f"[{satellite.shortName}] NaN fraction {nan_fraction:.1%} exceeds threshold {threshold:.1%} "
        f"(dateType={satellite.dateType!r})"
    )


def assert_single_image_quality(arr: np.ndarray, satellite: AbstractSatellite) -> None:
    """Assert that a single-image numpy array is non-empty with acceptable NaN fraction."""
    assert arr.size > 0, f"[{satellite.shortName}] Empty array returned"

    threshold = MAX_NAN_BY_DATE_TYPE.get(satellite.dateType, 0.05)
    nan_fraction = np.isnan(arr.astype(float)).mean()
    assert (
        nan_fraction <= threshold
    ), f"[{satellite.shortName}] NaN fraction {nan_fraction:.1%} exceeds threshold {threshold:.1%}"


def get_all_satellites_for_test() -> list[AbstractSatellite]:
    return [
        agl.sat.Sentinel2(),
        agl.sat.Sentinel2(use_sr=False),
        agl.sat.Sentinel1GRD(),
        agl.sat.Sentinel1GRD(ascending=False),
    ]


def get_all_date_types_for_test() -> list[str]:
    return ["doy", "year", "fyear"]


def get_all_reducers_for_test() -> list[str]:
    return ["min", "max", "mean", "median", "mode", "std", "var", "p2", "p98", "kurt", "skew"]
