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
    """MODIS Terra + Aqua daily merged — global coverage from 2000-02-24, 250 m resolution, daily cadence.

    Merges Terra (MOD09GQ) and Aqua (MYD09GQ) into a single collection so
    both overpasses contribute observations on the same day.

    Available bands: ``red``, ``nir``.

    .. warning::
        Daily MODIS has a high presence of residual clouds even after masking.
        For most analyses ``Modis8Days`` (8-day composites) produces cleaner data.

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands.  Defaults to ``{"red", "nir"}``.
    indices : set of str, optional
        Spectral indices to compute (e.g. ``{"ndvi"}``).
    use_cloud_mask : bool, default True
        Mask pixels flagged as cloudy by bit 10 of the ``state_1km`` QA band.
        Disabling this delivers more observations but severely increases noise.
    min_valid_pixel_count : int, default 2
        Images with fewer valid pixels over the ROI are discarded.
    border_pixels_to_erode : float, default 0.5
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 190_000
        Skip border erosion for geometries smaller than this area (m²).
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
        bands_: list[str] = sorted({"red", "nir"}) if bands is None else sorted(bands)

        indices_: list[str] = [] if indices is None else sorted(indices)

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

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands_)]

        self.selectedIndices = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices_)
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
            lambda img: ee.Image(img).addBands(ee.Image(img).add(ee.Number(100)).divide(ee.Number(16_100)), overwrite=True)
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
    """MODIS Terra + Aqua 8-day composites — global coverage from 2000-02-18, 250 m resolution.

    8-day best-pixel composites of Terra (MOD09Q1) and Aqua (MYD09Q1).
    Compositing reduces cloud contamination substantially compared to daily
    imagery, making this the recommended MODIS product for most time-series
    analyses.

    Available bands: ``red``, ``nir``.

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands.  Defaults to ``{"red", "nir"}``.
    indices : set of str, optional
        Spectral indices to compute (e.g. ``{"ndvi", "evi2"}``).
    use_cloud_mask : bool, default True
        Retain only pixels flagged as clear (bits 0–1 of the ``State`` QA band
        equal 00).  Mixed and cloudy states are discarded.
    min_valid_pixel_count : int, default 2
        Images with fewer valid pixels over the ROI are discarded.
    border_pixels_to_erode : float, default 0.5
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 190_000
        Skip border erosion for geometries smaller than this area (m²).
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
        bands_: list[str] = sorted({"red", "nir"}) if bands is None else sorted(bands)

        indices_: list[str] = [] if indices is None else sorted(indices)

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

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands_)]

        self.selectedIndices = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices_)
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
            lambda img: ee.Image(img).addBands(ee.Image(img).add(ee.Number(100)).divide(ee.Number(16_100)), overwrite=True)
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
