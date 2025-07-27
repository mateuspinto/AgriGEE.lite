from datetime import datetime
from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import OpticalSatellite


def extract_dates(imgcol):
    return imgcol.aggregate_array("ZZ_USER_TIME_DUMMY").distinct()


def filter_by_common_dates(ic, dates):
    return ic.filter(ee.Filter.inList("ZZ_USER_TIME_DUMMY", dates))


def intersect_lists(list1, list2):
    return list1.map(lambda el: ee.Algorithms.If(list2.contains(el), el, None)).removeAll([None])


def rename_bands(collection: ee.ImageCollection, prefix: str, postfix: str):
    def rename(image):
        image = ee.Image(image)
        band_names = image.bandNames()
        new_names = band_names.map(lambda name: ee.String(prefix).cat(name).cat(ee.String(postfix)))
        return image.select(band_names, new_names)

    return collection.map(rename)


class UnifiedSatellite(OpticalSatellite):
    def __init__(self, satellite_a: OpticalSatellite, satellite_b: OpticalSatellite):
        super().__init__()
        self.sat_a = satellite_a
        self.sat_b = satellite_b

        # Get the intersection between start date and end date of both satellites
        self.startDate = max(
            datetime.fromisoformat(satellite_a.startDate), datetime.fromisoformat(satellite_b.startDate)
        ).isoformat()
        self.endDate = min(
            datetime.fromisoformat(satellite_a.endDate), datetime.fromisoformat(satellite_b.endDate)
        ).isoformat()

        self.pixelSize = min(satellite_a.pixelSize, satellite_b.pixelSize)

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        sat_a = self.sat_a.imageCollection(ee_feature)
        sat_b = self.sat_b.imageCollection(ee_feature)

        sat_a_dates = extract_dates(sat_a)
        sat_b_dates = extract_dates(sat_b)

        common_dates = intersect_lists(sat_a_dates, sat_b_dates)

        sat_a_filtered = filter_by_common_dates(sat_a, common_dates)
        sat_b_filtered = filter_by_common_dates(sat_b, common_dates)

        sat_a_filtered = rename_bands(sat_a_filtered, "8", self.sat_a.shortName)
        sat_b_filtered = rename_bands(sat_b_filtered, "7", self.sat_b.shortName)

        merged = sat_a_filtered.linkCollection(
            sat_b_filtered, matchPropertyName="ZZ_USER_TIME_DUMMY", linkedBands=sat_b_filtered.first().bandNames()
        )

        return merged

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
