from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import DataSourceSatellite


class SatelliteEmbedding(DataSourceSatellite):
    """Google Satellite Embedding V1 — annual 64-dimensional embeddings from 2017 to 2023, ~10 m resolution.

    Pre-computed embeddings learned from multi-sensor satellite imagery by
    Google.  Each pixel encodes its spectral–temporal context into a 64-float
    vector, useful as input features for downstream ML models without having to
    engineer band combinations manually.

    For each annual image, ``compute()`` returns:
    - ``<band>_median`` — the embedding value at the geometry's centroid
      (preserves z-sphere normalisation).
    - ``<band>_stdDev`` — standard deviation of the embedding across the
      geometry interior (spatial heterogeneity signal).

    Parameters
    ----------
    bands : list of str, optional
        Subset of embedding dimensions (``"A00"`` to ``"A63"``).  Defaults
        to all 64.
    min_valid_pixel_count : int, default 1
        Images with fewer valid pixels over the ROI are discarded.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before stdDev extraction.
    min_area_to_keep_border : int, default 35_000
        Skip border erosion for geometries smaller than this area (m²).
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

        self.toDownloadSelectors = [
            f"{renamed}_median" for _, renamed in self.selectedBands
        ] + [
            f"{renamed}_stdDev" for _, renamed in self.selectedBands
        ]

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
