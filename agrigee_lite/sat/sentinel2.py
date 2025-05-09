from functools import partial

import ee

from agrigee_lite.ee_utils import (
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
        use_sr: bool = True,
        rescale_0_1: bool = True,
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

        remap_bands = {s: f"{(n + 10):02}_{s}" for n, s in enumerate(bands)}

        self.selectedBands: dict[str, str] = {
            remap_bands[band]: self.availableBands[band] for band in bands if band in self.availableBands
        }

        self.rescale_0_1 = rescale_0_1

        self.scaleBands = lambda x: x if rescale_0_1 else x / 10000

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()

        ee_start_date = ee_feature.get("s")
        ee_end_date = ee_feature.get("e")

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start_date, ee_end_date))

        s2_img = (
            ee.ImageCollection(self.imageCollectionName)
            .filter(ee_filter)
            .select(
                list(self.selectedBands.values()),
                list(self.selectedBands.keys()),
            )
        )

        s2_cloud_mask = (
            ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            .filter(ee_filter)
            .select(["cs_cdf"], ["cloud"])
        )

        s2_img = s2_img.combine(s2_cloud_mask)

        s2_img = s2_img.map(lambda img: ee_cloud_probability_mask(img, 0.7, True))
        s2_img = ee_filter_img_collection_invalid_pixels(s2_img, ee_geometry, self.pixelSize, 20)

        if self.rescale_0_1:
            s2_img = s2_img.map(lambda img: ee.Image(img).addBands(ee.Image(img).divide(10000), overwrite=True))

        return ee.ImageCollection(s2_img)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee_safe_remove_borders(ee_geometry, self.pixelSize, 35000)

        ee_geometry = ee.Geometry(
            ee.Algorithms.If(
                ee_geometry.buffer(-self.pixelSize).area().gte(35000),
                ee_geometry.buffer(-self.pixelSize),
                ee_geometry,
            )
        )
        s2_img = self.imageCollection(ee_feature)

        # round_int_16 is True only if reducers are None or contain exclusively 'mean' and/or 'median' and the image is not rescaled to 0-1
        allowed_reducers = {"mean", "median"}
        round_int_16 = (reducers is None or set(reducers).issubset(allowed_reducers)) and not self.rescale_0_1

        features = s2_img.map(
            partial(
                ee_map_bands_and_doy,
                ee_geometry=ee_geometry,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
                date_types=date_types,
                round_int_16=round_int_16,
            )
        )

        return features

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
