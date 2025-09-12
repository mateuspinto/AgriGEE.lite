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


class ModisDaily(OpticalSatellite):
    """
    ⚠️⚠️⚠️  Note: Despite this cloud mask, daily MODIS imagery tends to have **a high presence of residual clouds**. It is recommended to use Modis8Days for cleaner data. ⚠️⚠️⚠️

    Satellite abstraction for MODIS Terra and Aqua (Daily composites).

    MODIS (Moderate Resolution Imaging Spectroradiometer) is a key instrument aboard NASA's Terra and Aqua satellites,
    offering daily global coverage for environmental and land surface monitoring.

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['red', 'nir'].
    indices : set of str, optional
        List of spectral indices to compute from selected bands.
    use_cloud_mask : bool, default=True
        Whether to apply cloud masking using bit 10 of the 'state_1km' QA band.
        When set to False, no cloud filtering is applied (results may be ULTRA NOISY).
    min_valid_pixel_count : int, default=2
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=0.5
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=190_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Cloud Masking
    -------------
    Cloudy pixels are masked using bit 10 of the 'state_1km' QA band:
        - 0: clear
        - 1: cloudy

    Only pixels with bit 10 equal to 0 (clear) are retained.

    Satellite Information
    ---------------------
    +----------------------------+------------------------+
    | Field                      | Value                  |
    +----------------------------+------------------------+
    | Name                       | MODIS (Daily)          |
    | Platforms                  | Terra, Aqua            |
    | Temporal Resolution        | 1 day                  |
    | Pixel Size                 | 250 meters             |
    | Coverage                   | Global                 |
    +----------------------------+------------------------+

    Collection Dates
    ----------------
    +--------+------------+------------+
    | Source | Start Date | End Date  |
    +--------+------------+------------+
    | Terra  | 2000-02-24 | present   |
    | Aqua   | 2002-07-04 | present   |
    +--------+------------+------------+

    Band Information
    ----------------
    +-----------+----------------+----------------+------------------------+
    | Band Name | Original Band  | Resolution     | Spectral Wavelength    |
    +-----------+----------------+----------------+------------------------+
    | red       | sur_refl_b01   | 250 meters     | 620-670 nm             |
    | nir       | sur_refl_b02   | 250 meters     | 841-876 nm             |
    +-----------+----------------+----------------+------------------------+

    Notes
    -----
    Cloud Mask Reference (QA 'state_1km' band documentation):
        https://lpdaac.usgs.gov/documents/925/MOD09_User_Guide_V61.pdf

    MODIS Collections on Google Earth Engine:
        - Terra (MOD09GQ - reflectance): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GQ
        - Terra (MOD09GA - QA band):     https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09GA
        - Aqua  (MYD09GQ - reflectance): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD09GQ
        - Aqua  (MYD09GA - QA band):     https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD09GA
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 2,
        border_pixels_to_erode: float = 0.5,
        min_area_to_keep_border: int = 190_000,
    ) -> None:
        bands = sorted({"red", "nir"}) if bands is None else sorted(bands)

        indices = [] if indices is None else sorted(indices)

        super().__init__()

        self.shortName = "modis"
        self.pixelSize = 250
        self.startDate = "2000-02-24"
        self.endDate = "2050-01-01"

        self._terra_vis = "MODIS/061/MOD09GQ"
        self._terra_qa = "MODIS/061/MOD09GA"
        self._aqua_vis = "MODIS/061/MYD09GQ"
        self._aqua_qa = "MODIS/061/MYD09GA"

        self.availableBands = {
            "red": "sur_refl_b01",
            "nir": "sur_refl_b02",
        }

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
        ]

        self.useCloudMask = use_cloud_mask
        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.toDownloadSelectors = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

    @staticmethod
    def _mask_modis_clouds(img: ee.Image) -> ee.Image:
        """Bit-test bit 10 of *state_1km* (value 0 = clear)."""
        qa = img.select("state_1km")
        bit_mask = 1 << 10
        return img.updateMask(qa.bitwiseAnd(bit_mask).eq(0))

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        """
        Build the merged, cloud-masked Terra + Aqua collection *exactly*
        like the stand-alone helper did.
        """
        ee_geometry = ee_feature.geometry()
        ee_filter = ee.Filter.And(
            ee.Filter.bounds(ee_geometry),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        def _base(vis: str, qa: str) -> ee.ImageCollection:
            collection = ee.ImageCollection(vis).linkCollection(ee.ImageCollection(qa), ["state_1km"]).filter(ee_filter)
            if self.useCloudMask:
                collection = collection.map(self._mask_modis_clouds)

            return collection.select(
                list(self.availableBands.values()),
                list(self.availableBands.keys()),
            )

        terra = _base(self._terra_vis, self._terra_qa)
        aqua = _base(self._aqua_vis, self._aqua_qa)

        modis_imgc = terra.merge(aqua)

        modis_imgc = modis_imgc.map(
            lambda img: ee.Image(img).addBands(ee.Image(img).add(100).divide(16_100), overwrite=True)
        )

        if self.selectedIndices:
            modis_imgc = modis_imgc.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        modis_imgc = modis_imgc.select(
            [natural_band_name for natural_band_name, _ in self.selectedBands]
            + [indice_name for _, indice_name, _ in self.selectedIndices],
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

        modis_imgc = ee_filter_img_collection_invalid_pixels(
            modis_imgc, ee_geometry, self.pixelSize, self.minValidPixelCount
        )

        return ee.ImageCollection(modis_imgc)

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

        modis = self.imageCollection(ee_feature)

        feats = modis.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )
        return feats


class Modis8Days(OpticalSatellite):
    """
    Satellite abstraction for MODIS Terra and Aqua (8-day composites).

    MODIS (Moderate Resolution Imaging Spectroradiometer) is a key instrument aboard NASA's Terra and Aqua satellites,
    providing global coverage for land, ocean, and atmospheric monitoring at frequent intervals.

    Parameters
    ----------
    bands : list of str, optional
        List of bands to select. Defaults to ['red', 'nir'].
    indices : list of str, optional
        List of spectral indices to compute from selected bands.
    use_cloud_mask : bool, default=True
        Whether to apply a cloud mask based on the QA 'State' band (bits 0-1).
        If True, only pixels with cloud state == 0 (clear) are retained.
    min_valid_pixel_count : int, default=2
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=0.5
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=190_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Cloud Masking
    -------------
    Cloudy pixels are masked using bits 0-1 of the 'State' QA band, which encode cloud state:
        - 00: clear
        - 01: cloudy
        - 10: mixed
        - 11: not set

    The masking keeps only pixels with value 00 (clear) if `use_cloud_mask=True`.

    Satellite Information
    ---------------------
    +----------------------------+------------------------+
    | Field                      | Value                  |
    +----------------------------+------------------------+
    | Name                       | MODIS (8-day)          |
    | Platforms                  | Terra, Aqua            |
    | Temporal Resolution        | 8 days                 |
    | Pixel Size                 | 250 meters             |
    | Coverage                   | Global                 |
    +----------------------------+------------------------+

    Collection Dates
    ----------------
    +--------+------------+------------+
    | Source | Start Date | End Date  |
    +--------+------------+------------+
    | Terra  | 2000-02-18 | present   |
    | Aqua   | 2002-07-04 | present   |
    +--------+------------+------------+

    Band Information
    ----------------
    +-----------+----------------+----------------+------------------------+
    | Band Name | Original Band  | Resolution     | Spectral Wavelength    |
    +-----------+----------------+----------------+------------------------+
    | red       | sur_refl_b01   | 250 meters     | 620-670 nm             |
    | nir       | sur_refl_b02   | 250 meters     | 841-876 nm             |
    +-----------+----------------+----------------+------------------------+

    Notes
    -----
    Cloud Mask Reference (QA 'State' band documentation):
        https://lpdaac.usgs.gov/documents/925/MOD09_User_Guide_V61.pdf

    MODIS Collections on Google Earth Engine:
        - Terra (MOD09Q1): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD09Q1
        - Aqua  (MYD09Q1): https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD09Q1
    """

    def __init__(
        self,
        bands: list[str] | None = None,
        indices: list[str] | None = None,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 2,
        border_pixels_to_erode: float = 0.5,
        min_area_to_keep_border: int = 190_000,
    ) -> None:
        bands = sorted({"red", "nir"}) if bands is None else sorted(bands)

        indices = [] if indices is None else sorted(indices)

        super().__init__()

        self.shortName = "modis8days"
        self.pixelSize = 250
        self.startDate = "2000-02-18"
        self.endDate = "2050-01-01"

        self._terra = "MODIS/061/MOD09Q1"
        self._aqua = "MODIS/061/MYD09Q1"

        self.availableBands = {
            "red": "sur_refl_b01",
            "nir": "sur_refl_b02",
        }

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
        ]

        self.useCloudMask = use_cloud_mask
        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.toDownloadSelectors = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

    @staticmethod
    def _mask_modis8days_clouds(img: ee.Image) -> ee.Image:
        """Mask cloudy pixels based on bits 0-1 of 'State' QA band."""
        qa = img.select("State")
        cloud_state = qa.bitwiseAnd(3)  # 3 == 0b11
        return img.updateMask(cloud_state.eq(0))

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()

        if self.borderPixelsToErode != 0:
            ee_geometry = ee_safe_remove_borders(
                ee_geometry, round(self.borderPixelsToErode * self.pixelSize), self.minAreaToKeepBorder
            )
            ee_feature = ee_feature.setGeometry(ee_geometry)

        ee_filter = ee.Filter.And(
            ee.Filter.bounds(ee_geometry),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        def _base(path: str) -> ee.ImageCollection:
            collection = ee.ImageCollection(path).filter(ee_filter)
            if self.useCloudMask:
                collection = collection.map(self._mask_modis8days_clouds)

            return collection.select(
                list(self.availableBands.values()),
                list(self.availableBands.keys()),
            )

        terra = _base(self._terra)
        aqua = _base(self._aqua)

        modis_imgc = terra.merge(aqua)

        modis_imgc = modis_imgc.map(
            lambda img: ee.Image(img).addBands(ee.Image(img).add(100).divide(16_100), overwrite=True)
        )

        if self.selectedIndices:
            modis_imgc = modis_imgc.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        modis_imgc = modis_imgc.select(
            [natural_band_name for natural_band_name, _ in self.selectedBands]
            + [indice_name for _, indice_name, _ in self.selectedIndices],
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

        modis_imgc = ee_filter_img_collection_invalid_pixels(
            modis_imgc, ee_geometry, self.pixelSize, self.minValidPixelCount
        )

        return ee.ImageCollection(modis_imgc)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: set[str] | None = None,
    ) -> ee.FeatureCollection:
        geom = ee_feature.geometry()
        geom = ee_safe_remove_borders(geom, self.pixelSize // 2, 190_000)
        ee_feature = ee_feature.setGeometry(geom)

        modis = self.imageCollection(ee_feature)

        feats = modis.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(geom, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )
        return feats
