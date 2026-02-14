from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_add_indexes_to_image,
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
)
from agrigee_lite.sat.abstract_satellite import OpticalSatellite


class NAIP(OpticalSatellite):
    """
    Satellite abstraction for USDA NAIP (National Agriculture Imagery Program).

    The NAIP imagery is acquired during the agricultural growing seasons in the continental U.S.
    This class provides access to the DOQQ (Digital Orthophoto Quarter Quad) dataset.

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['blue', 'green', 'red', 'nir'].
    indices : set of str, optional
        Optical indices to compute (e.g., 'ndvi', 'ndwi'). Defaults to [].

    Satellite Information
    ---------------------
    +----------------------------+-------------------------------+
    | Field                      | Value                         |
    +----------------------------+-------------------------------+
    | Name                       | USDA NAIP                     |
    | Sensor                     | Optical (RGB + NIR)           |
    | Platform                   | Aircraft (Aerial)             |
    | Revisit Time               | Annual/Biennial (varies)      |
    | Pixel Size                 | 1 meter                       |
    | Coverage                   | Continental USA (CONUS)       |
    +----------------------------+-------------------------------+

    Band Information
    ----------------
    +-----------+---------+------------+-------------------------------------------+
    | Band Name | Type    | Resolution | Description                               |
    +-----------+---------+------------+-------------------------------------------+
    | red       | Optical | 1 m        | Red                                       |
    | green     | Optical | 1 m        | Green                                     |
    | blue      | Optical | 1 m        | Blue                                      |
    | nir       | Optical | 1 m        | Near-Infrared                             |
    +-----------+---------+------------+-------------------------------------------+

    Notes
    -----
    - Earth Engine Dataset:
        https://developers.google.com/earth-engine/datasets/catalog/USDA_NAIP_DOQQ
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
    ):
        bands = sorted({"blue", "green", "red", "nir"}) if bands is None else sorted(bands)
        indices = [] if indices is None else sorted(indices)

        super().__init__()

        self.imageCollectionName: str = "USDA/NAIP/DOQQ"
        self.pixelSize: int = 1
        self.startDate: str = "2003-01-01"
        self.endDate: str = "2050-01-01"
        self.shortName: str = "naip"

        self.availableBands: dict[str, str] = {"blue": "B", "green": "G", "red": "R", "nir": "N"}

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
        ]

        self.toDownloadSelectors = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_start = ee_feature.get("s")
        ee_end = ee_feature.get("e")

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start, ee_end))

        naip_img = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        naip_img = naip_img.select(
            [self.availableBands[b] for b, _ in self.selectedBands],
            [b for b, _ in self.selectedBands],
        )

        if self.selectedIndices:
            naip_img = naip_img.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        src_names = [band_name for band_name, _ in self.selectedBands] + [
            indice_name for _, indice_name, _ in self.selectedIndices
        ]

        dst_names = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

        naip_img = naip_img.select(src_names, dst_names)

        naip_img = ee_filter_img_collection_invalid_pixels(naip_img, ee_geometry, self.pixelSize, 0)

        return naip_img

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: set[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()

        naip_img = self.imageCollection(ee_feature)

        features = naip_img.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )

        return features
