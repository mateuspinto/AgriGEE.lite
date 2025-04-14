import numpy as np


def np_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red)


def np_gndvi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - green) / (nir + green)


def np_ndwi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    return (nir - swir1) / (nir + swir1)


def np_savi(red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
    return ((nir - red) / (nir + red + L)) * (1 + L)


def np_evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1)


def np_evi2(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return 2.5 * (nir - red) / (nir + 2.4 * red + 1)


def np_msavi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (2 * nir + 1 - np.sqrt((2 * nir + 1) ** 2 - 8 * (nir - red))) / 2


def np_ndre(red_edge: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - red_edge) / (nir + red_edge)


def np_mcari(green: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return ((nir - red) - 0.2 * (nir - green)) * (nir / red)


def np_gci(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir / green) - 1


def np_bsi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    return ((swir1 + red) - (nir + blue)) / ((swir1 + red) + (nir + blue))


def np_ci_red(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir / red) - 1


def np_ci_green(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir / green) - 1


def np_osavi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - red) / (nir + red + 0.16)


def np_arvi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    return (nir - (2 * red - blue)) / (nir + (2 * red - blue))


def np_vhvv(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    return vh / vv


ALL_NUMPY_INDICES = {
    "ndvi": np_ndvi,
    "gndvi": np_gndvi,
    "ndwi": np_ndwi,
    "savi": np_savi,
    "evi": np_evi,
    "evi2": np_evi2,
    "msavi": np_msavi,
    "ndre": np_ndre,
    "mcari": np_mcari,
    "gci": np_gci,
    "bsi": np_bsi,
    "ci_red": np_ci_red,
    "ci_green": np_ci_green,
    "osavi": np_osavi,
    "arvi": np_arvi,
    "vhvv": np_vhvv,
}
