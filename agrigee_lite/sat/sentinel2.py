from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_add_indexes_to_image,
    ee_cloud_probability_mask,
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import OpticalSatellite


class Sentinel2(OpticalSatellite):
    """
    Satellite abstraction for Sentinel-2 (HARMONIZED collections).

    Sentinel-2 is a constellation of twins Earth observation satellites,
    operated by ESA, designed for land monitoring, vegetation, soil, water cover, and coastal areas.

    Parameters
    ----------
    bands : list of str, optional
        List of bands to select. Defaults to all 10 bands most used for vegetation and soil analysis.
    use_sr : bool, defaults to False
        If True, uses surface reflectance (BOA, 'S2_SR_HARMONIZED').
        If False, uses top-of-atmosphere reflectance ('S2_HARMONIZED').

    Satellite Information
    ---------------------
    +-----------------------------------+------------------------+
    | Field                             | Value                  |
    +-----------------------------------+------------------------+
    | Name                              | Sentinel-2             |
    | Revisit Time                      | 5 days                 |
    | Revisit Time (cloud-free estimate) | ~7 days               |
    | Pixel Size                        | 10 meters              |
    | Coverage                          | Global                 |
    +-----------------------------------+------------------------+

    Collection Dates
    ----------------
    +------------------+------------+----------------+
    | Collection Type  | Start Date | End Date        |
    +------------------+------------+----------------+
    | TOA (Top of Atmosphere) | 2016-01-01 | present    |
    | SR (Surface Reflectance) | 2019-01-01 | present   |
    +------------------+------------+----------------+

    Band Information
    ------------
    +------------+---------------+--------------+----------------------+
    | Band Name  | Original Band | Resolution   | Spectral Wavelength   |
    +------------+---------------+--------------+----------------------+
    | blue       | B2            | 10 m        | 492 nm                |
    | green      | B3            | 10 m        | 559 nm                |
    | red        | B4            | 10 m        | 665 nm                |
    | re1        | B5            | 20 m        | 704 nm                |
    | re2        | B6            | 20 m        | 739 nm                |
    | re3        | B7            | 20 m        | 780 nm                |
    | nir        | B8            | 10 m        | 833 nm                |
    | re4        | B8A           | 20 m        | 864 nm                |
    | swir1      | B11           | 20 m        | 1610 nm               |
    | swir2      | B12           | 20 m        | 2186 nm               |
    +------------+---------------+--------------+----------------------+
    """

    def __init__(
        self,
        bands: list[str] | None = None,
        indices: list[str] | None = None,
        use_sr: bool = True,
    ):
        if bands is None:
            bands = [
                "blue",
                "green",
                "red",
                "re1",
                "re2",
                "re3",
                "nir",
                "re4",
                "swir1",
                "swir2",
            ]

        if indices is None:
            indices = []

        super().__init__()
        self.useSr = use_sr
        self.imageCollectionName = "COPERNICUS/S2_SR_HARMONIZED" if use_sr else "COPERNICUS/S2_HARMONIZED"
        self.pixelSize: int = 10

        self.startDate: str = "2019-01-01" if use_sr else "2016-01-01"
        self.endDate: str = "2050-01-01"
        self.shortName: str = "s2sr" if use_sr else "s2"

        self.availableBands: dict[str, str] = {
            "blue": "B2",
            "green": "B3",
            "red": "B4",
            "re1": "B5",
            "re2": "B6",
            "re3": "B7",
            "nir": "B8",
            "re4": "B8A",
            "swir1": "B11",
            "swir2": "B12",
        }

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
        ]

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()

        ee_start_date = ee_feature.get("s")
        ee_end_date = ee_feature.get("e")

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start_date, ee_end_date))

        s2_img = (
            ee.ImageCollection(self.imageCollectionName)
            .filter(ee_filter)
            .select(
                list(self.availableBands.values()),
                list(self.availableBands.keys()),
            )
        )

        s2_img = s2_img.map(lambda img: ee.Image(img).addBands(ee.Image(img).divide(10000), overwrite=True))

        if self.selectedIndices:
            s2_img = s2_img.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        s2_img = s2_img.select(
            [natural_band_name for natural_band_name, _ in self.selectedBands]
            + [indice_name for _, indice_name, _ in self.selectedIndices],
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

        s2_cloud_mask = (
            ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            .filter(ee_filter)
            .select(["cs_cdf"], ["cloud"])
        )

        s2_img = s2_img.combine(s2_cloud_mask)

        s2_img = s2_img.map(lambda img: ee_cloud_probability_mask(img, 0.7, True))
        s2_img = ee_filter_img_collection_invalid_pixels(s2_img, ee_geometry, self.pixelSize, 20)

        return ee.ImageCollection(s2_img)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee_safe_remove_borders(ee_geometry, self.pixelSize, 35000)
        ee_feature = ee_feature.setGeometry(ee_geometry)

        s2_img = self.imageCollection(ee_feature)

        features = s2_img.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )

        return features
