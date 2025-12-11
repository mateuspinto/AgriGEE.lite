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
    """
    Satellite abstraction for NASA HLS (Harmonized Landsat Sentinel-2) - Sentinel-2 component.

    The Harmonized Landsat Sentinel-2 (HLS) project provides consistent surface reflectance (SR) data
    from the Operational Land Imager (OLI) aboard Landsat-8/9 and the Multi-Spectral Instrument (MSI)
    aboard Sentinel-2A/B. This class specifically handles the Sentinel-2 component (HLSS30 v002).

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'].
    indices : set of str, optional
        Spectral indices to compute from the selected bands (e.g., 'ndvi', 'evi').
    use_quality_mask : bool, default=True
        Whether to apply the Fmask quality mask to filter clouds, shadows, snow, etc.
    min_valid_pixel_count : int, default=20
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=35_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Quality Masking (Fmask)
    -----------------------
    When `use_quality_mask=True`, the `Fmask` band is used to filter out invalid pixels:
        - Bit 0: Cirrus (reserved, not used)
        - Bit 1: Cloud (0=No, 1=Yes)
        - Bit 2: Adjacent to cloud/shadow (0=No, 1=Yes)
        - Bit 3: Cloud shadow (0=No, 1=Yes)
        - Bit 4: Snow/ice (0=No, 1=Yes)
        - Bit 5: Water (0=No, 1=Yes)
        - Bits 6-7: Aerosol level (0=Climatology, 1=Low, 2=Moderate, 3=High)

    Satellite Information
    ---------------------
    +------------------------------------+------------------------+
    | Field                              | Value                  |
    +------------------------------------+------------------------+
    | Name                               | HLS Sentinel-2         |
    | Sensor                             | MSI (Sentinel-2A/B)    |
    | Revisit Time                       | ~2-3 days (combined)   |
    | Pixel Size                         | 30 meters              |
    | Coverage                           | Global                 |
    +------------------------------------+------------------------+

    Collection Dates
    ----------------
    +----------------------------+------------+------------+
    | Collection Type            | Start Date | End Date   |
    +----------------------------+------------+------------+
    | HLSS30 v002                | 2015-11-30 | present    |
    +----------------------------+------------+------------+

    Band Information
    ----------------
    +-----------+---------------+--------------+------------------------+
    | Band Name | Original Band | Resolution   | Spectral Wavelength    |
    +-----------+---------------+--------------+------------------------+
    | coastal   | B1            | 30 m         | 443 nm                 |
    | blue      | B2            | 30 m         | 482 nm                 |
    | green     | B3            | 30 m         | 561 nm                 |
    | red       | B4            | 30 m         | 655 nm                 |
    | re1       | B5            | 30 m         | 865 nm                 |
    | re2       | B6            | 30 m         | 1609 nm                |
    | re3       | B7            | 30 m         | 2201 nm                |
    | nir       | B8            | 30 m         | 833 nm                 |
    | re4       | B8A           | 30 m         | 865 nm                 |
    | swir1     | B11           | 30 m         | 1609 nm                |
    | swir2     | B12           | 30 m         | 2201 nm                |
    +-----------+---------------+--------------+------------------------+

    Notes
    -----
    - Earth Engine Dataset:
        https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSS30_v002

    - Fmask Quality Band Documentation:
        https://lpdaac.usgs.gov/documents/1326/HLS_User_Guide_V2.pdf

    - HLS provides atmospherically corrected surface reflectance (SR) data that is
      harmonized across Landsat and Sentinel-2 missions for consistent time series analysis.
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
        bands = (
            sorted({"blue", "green", "red", "nir", "swir1", "swir2"})
            if bands is None
            else sorted(bands)
        )

        indices = [] if indices is None else sorted(indices)

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

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
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

        hls_img = ee_filter_img_collection_invalid_pixels(
            hls_img, ee_geometry, self.pixelSize, self.minValidPixelCount
        )

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
    """
    Satellite abstraction for NASA HLS (Harmonized Landsat Sentinel-2) - Landsat component.

    The Harmonized Landsat Sentinel-2 (HLS) project provides consistent surface reflectance (SR) data
    from the Operational Land Imager (OLI) aboard Landsat-8/9 and the Multi-Spectral Instrument (MSI)
    aboard Sentinel-2A/B. This class specifically handles the Landsat component (HLSL30 v002).

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'].
    indices : set of str, optional
        Spectral indices to compute from the selected bands (e.g., 'ndvi', 'evi').
    use_quality_mask : bool, default=True
        Whether to apply the Fmask quality mask to filter clouds, shadows, snow, etc.
    min_valid_pixel_count : int, default=20
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=35_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Quality Masking (Fmask)
    -----------------------
    When `use_quality_mask=True`, the `Fmask` band is used to filter out invalid pixels:
        - Bit 0: Cirrus (reserved, not used)
        - Bit 1: Cloud (0=No, 1=Yes)
        - Bit 2: Adjacent to cloud/shadow (0=No, 1=Yes)
        - Bit 3: Cloud shadow (0=No, 1=Yes)
        - Bit 4: Snow/ice (0=No, 1=Yes)
        - Bit 5: Water (0=No, 1=Yes)
        - Bits 6-7: Aerosol level (0=Climatology, 1=Low, 2=Moderate, 3=High)

    Satellite Information
    ---------------------
    +------------------------------------+------------------------+
    | Field                              | Value                  |
    +------------------------------------+------------------------+
    | Name                               | HLS Landsat            |
    | Sensor                             | OLI (Landsat-8/9)      |
    | Revisit Time                       | ~2-3 days (combined)   |
    | Pixel Size                         | 30 meters              |
    | Coverage                           | Global                 |
    +------------------------------------+------------------------+

    Collection Dates
    ----------------
    +----------------------------+------------+------------+
    | Collection Type            | Start Date | End Date   |
    +----------------------------+------------+------------+
    | HLSL30 v002                | 2013-04-11 | present    |
    +----------------------------+------------+------------+

    Band Information
    ----------------
    +-----------+---------------+--------------+------------------------+
    | Band Name | Original Band | Resolution   | Spectral Wavelength    |
    +-----------+---------------+--------------+------------------------+
    | coastal   | B1            | 30 m         | 443 nm                 |
    | blue      | B2            | 30 m         | 482 nm                 |
    | green     | B3            | 30 m         | 561 nm                 |
    | red       | B4            | 30 m         | 655 nm                 |
    | nir       | B5            | 30 m         | 865 nm                 |
    | swir1     | B6            | 30 m         | 1609 nm                |
    | swir2     | B7            | 30 m         | 2201 nm                |
    | tirs1     | B10           | 30 m         | 1373 nm                |
    | tirs2     | B11           | 30 m         | 2196 nm                |
    +-----------+---------------+--------------+------------------------+

    Notes
    -----
    - Earth Engine Dataset:
        https://developers.google.com/earth-engine/datasets/catalog/NASA_HLS_HLSL30_v002

    - Fmask Quality Band Documentation:
        https://lpdaac.usgs.gov/documents/1326/HLS_User_Guide_V2.pdf

    - HLS provides atmospherically corrected surface reflectance (SR) data that is
      harmonized across Landsat and Sentinel-2 missions for consistent time series analysis.
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
        bands = (
            sorted({"blue", "green", "red", "nir", "swir1", "swir2", "tirs1", "tirs2"})
            if bands is None
            else sorted(bands)
        )

        indices = [] if indices is None else sorted(indices)

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

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
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

        hls_img = ee_filter_img_collection_invalid_pixels(
            hls_img, ee_geometry, self.pixelSize, self.minValidPixelCount
        )

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
