from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import DataSourceSatellite


class SatelliteEmbedding(DataSourceSatellite):
    """
    Satellite abstraction for the Google Satellite Embedding collection.

    This collection contains annual, manually curated embeddings derived from multi-sensor satellite data.

    IMPORTANT: It always returns the center point value as the median (in order to maintain z-sphere normalization) and the standard deviation of the geometry without the borders.

    Parameters
    ----------
    bands : list of str, optional
        List of bands to select. Defaults to all 64 embeddings.
    min_valid_pixel_count : int, default=1
        Minimum number of valid (non-eroded) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=35_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Satellite Information
    ---------------------
    +-----------------------------+-----------------------+
    | Field                       | Value                 |
    +-----------------------------+-----------------------+
    | Name                        | Satellite Embedding   |
    | Embedding Dimensions        | 64 (A0 to A63)        |
    | Pixel Size                  | ~10 meters            |
    | Temporal Resolution         | Annual                |
    | Coverage                    | Global                |
    +-----------------------------+-----------------------+

    Collection Dates
    ----------------
    +------------+------------+
    | Start Date | End Date   |
    +------------+------------+
    | 2017-01-01 | 2024-01-02 |
    +------------+------------+

    Notes
    ----------------
    Satellite Embedding V1:
        - Dataset: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL?hl=pt-br#bands
    """

    def __init__(
        self,
        bands: list[str] | None = None,
        min_valid_pixel_count: int = 1,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 35000,
    ):
        super().__init__()

        if bands is None:
            bands = [f"A{i:02}" for i in range(64)]

        self.imageCollectionName: str = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
        self.pixelSize: int = 10
        self.startDate: str = "2017-01-01"
        self.endDate: str = "2024-01-01"
        self.shortName: str = "satembed"

        self.availableBands: dict[str, str] = {b: b for b in bands}
        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_start = ee_feature.get("s")
        ee_end = ee_feature.get("e")

        ee_filter = ee.Filter.And(
            ee.Filter.bounds(ee_geometry),
            ee.Filter.date(ee_start, ee_end),
        )

        imgcol = (
            ee.ImageCollection(self.imageCollectionName)
            .filter(ee_filter)
            .select(
                list(self.availableBands.values()),
                list(self.availableBands.keys()),
            )
        )

        imgcol = imgcol.select(
            [natural for natural, _ in self.selectedBands],
            [renamed for _, renamed in self.selectedBands],
        )

        imgcol = ee_filter_img_collection_invalid_pixels(imgcol, ee_geometry, self.pixelSize, self.minValidPixelCount)

        return imgcol

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: set[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()

        if self.borderPixelsToErode != 0:
            ee_geometry = ee_safe_remove_borders(
                ee_geometry, round(self.borderPixelsToErode * self.pixelSize), self.minAreaToKeepBorder
            )
            ee_feature = ee_feature.setGeometry(ee_geometry)

        imgcol = self.imageCollection(ee_feature)

        def compute_stats(
            ee_img: ee.Image, ee_feature: ee.Feature, pixel_size: int, subsampling_max_pixels: ee.Number
        ) -> ee.Feature:
            ee_img = ee.Image(ee_img)

            median = ee_img.reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=ee_feature.geometry().centroid(0.001),
                scale=pixel_size,
                maxPixels=subsampling_max_pixels,
                bestEffort=True,
            )

            stddev = ee_img.reduceRegion(
                reducer=ee.Reducer.stdDev(),
                geometry=ee_geometry,
                scale=pixel_size,
                maxPixels=subsampling_max_pixels,
                bestEffort=True,
            )

            stddev = stddev.rename(stddev.keys(), stddev.keys().map(lambda k: ee.String(k).cat("_stdDev")), True)
            median = median.rename(median.keys(), median.keys().map(lambda k: ee.String(k).cat("_median")), True)

            props = ee.Dictionary(median).combine(stddev)

            props = props.set("00_indexnum", ee_feature.get("0"))
            props = props.set("01_timestamp", ee.Date(ee_img.date()).format("YYYY-MM-dd"))
            props = props.set("99_validPixelsCount", ee_img.get("ZZ_USER_VALID_PIXELS"))

            return ee.Feature(None, props)

        features = imgcol.map(
            partial(
                compute_stats,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
            )
        )

        return ee.FeatureCollection(features)
