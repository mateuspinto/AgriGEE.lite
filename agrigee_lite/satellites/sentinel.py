from functools import partial

import ee

from agrigee_lite.ee_utils import ee_cloud_probability_mask, ee_map_bands_and_doy, ee_map_valid_pixels
from agrigee_lite.satellites.abstract_satellite import AbstractSatellite


class Sentinel2(AbstractSatellite):
    def __init__(self, use_sr: bool = False):
        super().__init__()
        self.useSr = use_sr
        self.imageCollectionName = "COPERNICUS/S2_SR_HARMONIZED" if use_sr else "COPERNICUS/S2_HARMONIZED"
        self.startDate: str = "2015-06-23"
        self.endDate: str = ""
        self.originalBands: list[str] = [
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B11",
            "B12",
        ]
        self.renamedBands: list[str] = [
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

    def compute(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()

        ee_start_date = ee_feature.get("start_date")
        ee_end_date = ee_feature.get("end_date")

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start_date, ee_end_date))

        s2_img = (
            ee.ImageCollection(self.imageCollectionName)
            .filter(ee_filter)
            .select(
                self.originalBands,
                self.renamedBands,
            )
        )

        s2_cloud_mask = (
            ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
            .filter(ee_filter)
            .select(["cs_cdf"], ["cloud"])
        )

        s2_img = s2_img.combine(s2_cloud_mask)

        s2_img = s2_img.map(lambda img: ee_cloud_probability_mask(img, 0.7, True))
        s2_img = s2_img.map(lambda img: ee_map_valid_pixels(img, ee_geometry, 10)).filter(
            ee.Filter.gte("ZZ_USER_VALID_PIXELS", 20)
        )

        s2_img = (
            s2_img.map(lambda img: img.set("ZZ_USER_TIME_DUMMY", img.date().format("YYYY-MM-dd")))
            .sort("ZZ_USER_TIME_DUMMY")
            .distinct("ZZ_USER_TIME_DUMMY")
        )

        features = s2_img.map(
            partial(ee_map_bands_and_doy, ee_geometry=ee_geometry, ee_feature=ee_feature, scale=10, round_int_16=True)
        )

        return features
