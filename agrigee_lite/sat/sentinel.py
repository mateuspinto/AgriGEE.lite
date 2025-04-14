from functools import partial

import ee

from agrigee_lite.ee_utils import ee_cloud_probability_mask, ee_map_bands_and_doy, ee_map_valid_pixels
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


class Sentinel2(AbstractSatellite):
    def __init__(
        self,
        selected_bands: list[str] | None = None,
        use_sr: bool = False,
    ):
        if selected_bands is None:
            selected_bands = [
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

        self.startDate: str = "2019-01-01" if use_sr else "2016-01-01"
        self.endDate: str = ""
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

        remap_bands = {s: f"{(n + 10):02}_{s}" for n, s in enumerate(selected_bands)}

        self.selectedBands: dict[str, str] = {
            remap_bands[band]: self.availableBands[band] for band in selected_bands if band in self.availableBands
        }

        self.scaleBands = lambda x: x / 10000

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()

        ee_start_date = ee_feature.get("start_date")
        ee_end_date = ee_feature.get("end_date")

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
        s2_img = s2_img.map(lambda img: ee_map_valid_pixels(img, ee_geometry, 10)).filter(
            ee.Filter.gte("ZZ_USER_VALID_PIXELS", 20)
        )

        s2_img = (
            s2_img.map(lambda img: img.set("ZZ_USER_TIME_DUMMY", img.date().format("YYYY-MM-dd")))
            .sort("ZZ_USER_TIME_DUMMY")
            .distinct("ZZ_USER_TIME_DUMMY")
        )

        return ee.ImageCollection(s2_img)

    def compute(self, ee_feature: ee.Feature) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()

        s2_img = self.imageCollection(ee_feature)

        features = s2_img.map(
            partial(ee_map_bands_and_doy, ee_geometry=ee_geometry, ee_feature=ee_feature, scale=10, round_int_16=True)
        )

        return features
