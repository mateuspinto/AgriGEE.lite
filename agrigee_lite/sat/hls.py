from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_add_indexes_to_image,
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import OpticalSatellite


class HLSSentinel2(OpticalSatellite):
    """HLS Sentinel-2 (HLSS30 v002) — harmonised SR data from 2015-11-30, 30 m resolution, ~2–3 day revisit.

    NASA's Harmonized Landsat Sentinel-2 (HLS) project resamples Sentinel-2 to
    30 m and applies cross-calibration so its reflectance values are consistent
    with Landsat.  The main benefit is a denser time series (~2–3 days) from
    combining both constellations via ``TwoSatelliteFusion``.

    Available bands: ``coastal``, ``blue``, ``green``, ``red``, ``re1``,
    ``re2``, ``re3``, ``nir``, ``re4``, ``swir1``, ``swir2``.

    Quality masking uses the Fmask band (bits 1–4: cloud, adjacent-to-cloud,
    shadow, snow).

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands.  Defaults to all eleven.
    indices : set of str, optional
        Spectral indices to compute (e.g. ``{"ndvi", "evi2"}``).
    use_quality_mask : bool, default True
        Apply Fmask quality filtering.  Disabling delivers more images but with
        cloud and shadow contamination.
    min_valid_pixel_count : int, default 20
        Images below this valid-pixel count over the ROI are discarded.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 35_000
        Skip border erosion for geometries smaller than this area (m²).
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_quality_mask: bool = True,
        min_valid_pixel_count: int = 20,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 35000,
    ):
        super().__init__()

        self.imageCollectionName: str = "NASA/HLS/HLSS30/v002"
        self.pixelSize: int = 30
        self.startDate: str = "2015-11-30"
        self.endDate: str = "2050-01-01"
        self.shortName: str = "hls_s2"

        self.availableBands: dict[str, str] = {
            "coastal": "B1",
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

        bands_: list[str] = (
            ["coastal", "blue", "green", "red", "re1", "re2", "re3", "nir", "re4", "swir1", "swir2"]
            if bands is None
            else sorted(bands)
        )

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands_)]

        indices_: list[str] = [] if indices is None else sorted(indices)

        self.selectedIndices = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices_)
        ]

        self.use_quality_mask = use_quality_mask
        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.toDownloadSelectors = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

    @staticmethod
    def _mask_fmask(img: ee.Image) -> ee.Image:
        """
        Apply Fmask quality mask to exclude clouds, shadows, snow, and adjacent pixels.

        Fmask bit interpretation:
            Bit 1: Cloud (0=No, 1=Yes)
            Bit 2: Adjacent to cloud/shadow (0=No, 1=Yes)
            Bit 3: Cloud shadow (0=No, 1=Yes)
            Bit 4: Snow/ice (0=No, 1=Yes)

        Parameters
        ----------
        img : ee.Image

        Returns
        -------
        ee.Image
        """
        fmask = img.select("Fmask")

        # Create masks for each quality issue
        cloud = fmask.bitwiseAnd(1 << 1)  # Bit 1: Cloud
        adjacent = fmask.bitwiseAnd(1 << 2)  # Bit 2: Adjacent to cloud/shadow
        shadow = fmask.bitwiseAnd(1 << 3)  # Bit 3: Cloud shadow
        snow = fmask.bitwiseAnd(1 << 4)  # Bit 4: Snow/ice

        # Combine all masks - keep pixels where all bits are 0 (clear)
        clear_mask = cloud.Or(adjacent).Or(shadow).Or(snow).eq(0)

        return img.updateMask(clear_mask)

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_start = ee_feature.get("s")
        ee_end = ee_feature.get("e")

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start, ee_end))

        hls_img = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        if self.use_quality_mask:
            hls_img = hls_img.map(self._mask_fmask)

        hls_img = hls_img.select(
            [self.availableBands[b] for b, _ in self.selectedBands], [b for b, _ in self.selectedBands]
        )

        if self.selectedIndices:
            hls_img = hls_img.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        hls_img = hls_img.select(
            [natural_band_name for natural_band_name, _ in self.selectedBands]
            + [indice_name for _, indice_name, _ in self.selectedIndices],
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

        hls_img = ee_filter_img_collection_invalid_pixels(hls_img, ee_geometry, self.pixelSize, self.minValidPixelCount)

        return hls_img

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

        hls_img = self.imageCollection(ee_feature)

        features = hls_img.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )

        return features


class HLSLandsat(OpticalSatellite):
    """HLS Landsat (HLSL30 v002) — harmonised SR data from 2013-04-11, 30 m resolution, ~2–3 day revisit.

    Landsat 8/9 reprocessed by NASA's HLS project to be spectrally consistent
    with HLS Sentinel-2.  Use both together (via ``TwoSatelliteFusion``) to
    achieve a dense, cross-calibrated time series at 30 m.

    Available bands: ``coastal``, ``blue``, ``green``, ``red``, ``nir``,
    ``swir1``, ``swir2``, ``tirs1``, ``tirs2``.

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands.  Defaults to all nine.
    indices : set of str, optional
        Spectral indices to compute (e.g. ``{"ndvi"}``).
    use_quality_mask : bool, default True
        Apply Fmask quality filtering (bits 1–4: cloud, adjacent, shadow, snow).
    min_valid_pixel_count : int, default 20
        Images below this valid-pixel count over the ROI are discarded.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 35_000
        Skip border erosion for geometries smaller than this area (m²).
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_quality_mask: bool = True,
        min_valid_pixel_count: int = 20,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 35000,
    ):
        super().__init__()

        self.imageCollectionName: str = "NASA/HLS/HLSL30/v002"
        self.pixelSize: int = 30
        self.startDate: str = "2013-04-11"
        self.endDate: str = "2050-01-01"
        self.shortName: str = "hls_l8"

        self.availableBands: dict[str, str] = {
            "coastal": "B1",
            "blue": "B2",
            "green": "B3",
            "red": "B4",
            "nir": "B5",
            "swir1": "B6",
            "swir2": "B7",
            "tirs1": "B10",
            "tirs2": "B11",
        }

        bands_: list[str] = (
            ["coastal", "blue", "green", "red", "nir", "swir1", "swir2", "tirs1", "tirs2"]
            if bands is None
            else sorted(bands)
        )

        indices_: list[str] = [] if indices is None else sorted(indices)

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands_)]

        self.selectedIndices = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices_)
        ]

        self.use_quality_mask = use_quality_mask
        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.toDownloadSelectors = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

    @staticmethod
    def _mask_fmask(img: ee.Image) -> ee.Image:
        """
        Apply Fmask quality mask to exclude clouds, shadows, snow, and adjacent pixels.

        Fmask bit interpretation:
            Bit 1: Cloud (0=No, 1=Yes)
            Bit 2: Adjacent to cloud/shadow (0=No, 1=Yes)
            Bit 3: Cloud shadow (0=No, 1=Yes)
            Bit 4: Snow/ice (0=No, 1=Yes)

        Parameters
        ----------
        img : ee.Image

        Returns
        -------
        ee.Image
        """
        fmask = img.select("Fmask")

        # Create masks for each quality issue
        cloud = fmask.bitwiseAnd(1 << 1)  # Bit 1: Cloud
        adjacent = fmask.bitwiseAnd(1 << 2)  # Bit 2: Adjacent to cloud/shadow
        shadow = fmask.bitwiseAnd(1 << 3)  # Bit 3: Cloud shadow
        snow = fmask.bitwiseAnd(1 << 4)  # Bit 4: Snow/ice

        # Combine all masks - keep pixels where all bits are 0 (clear)
        clear_mask = cloud.Or(adjacent).Or(shadow).Or(snow).eq(0)

        return img.updateMask(clear_mask)

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_start = ee_feature.get("s")
        ee_end = ee_feature.get("e")

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start, ee_end))

        hls_img = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        if self.use_quality_mask:
            hls_img = hls_img.map(self._mask_fmask)

        hls_img = hls_img.select(
            [self.availableBands[b] for b, _ in self.selectedBands], [b for b, _ in self.selectedBands]
        )

        if self.selectedIndices:
            hls_img = hls_img.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        hls_img = hls_img.select(
            [natural_band_name for natural_band_name, _ in self.selectedBands]
            + [indice_name for _, indice_name, _ in self.selectedIndices],
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

        hls_img = ee_filter_img_collection_invalid_pixels(hls_img, ee_geometry, self.pixelSize, self.minValidPixelCount)

        return hls_img

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

        hls_img = self.imageCollection(ee_feature)

        features = hls_img.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )

        return features
