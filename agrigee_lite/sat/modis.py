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


class Modis(OpticalSatellite):
    def __init__(self, bands: list[str] | None = None, indices: list[str] | None = None) -> None:
        if bands is None:
            bands = ["red", "nir"]

        if indices is None:
            indices = []

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
            return (
                ee.ImageCollection(vis)
                .linkCollection(ee.ImageCollection(qa), ["state_1km"])
                .filter(ee_filter)
                .map(self._mask_modis_clouds)
                .select(
                    list(self.availableBands.values()),
                    list(self.availableBands.keys()),
                )
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

        modis_imgc = ee_filter_img_collection_invalid_pixels(modis_imgc, ee_geometry, self.pixelSize, 2)

        return ee.ImageCollection(modis_imgc)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
    ) -> ee.FeatureCollection:
        """Sample time series of median reflectance within *ee_feature*."""
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


class Modis8Days(OpticalSatellite):
    def __init__(self, bands: list[str] | None = None, indices: list[str] | None = None) -> None:
        if bands is None:
            bands = ["red", "nir"]

        if indices is None:
            indices = []

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

    @staticmethod
    def _mask_modis8days_clouds(img: ee.Image) -> ee.Image:
        """Mask cloudy pixels based on bits 0-1 of 'State' QA band."""
        qa = img.select("State")
        cloud_state = qa.bitwiseAnd(3)  # 3 == 0b11 → isola os bits 0 e 1
        return img.updateMask(cloud_state.eq(0))

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_filter = ee.Filter.And(
            ee.Filter.bounds(ee_geometry),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        def _base(path: str) -> ee.ImageCollection:
            return (
                ee.ImageCollection(path)
                .filter(ee_filter)
                .map(self._mask_modis8days_clouds)
                .select(
                    list(self.availableBands.values()),
                    list(self.availableBands.keys()),
                )
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

        modis_imgc = ee_filter_img_collection_invalid_pixels(modis_imgc, ee_geometry, self.pixelSize, 2)

        return ee.ImageCollection(modis_imgc)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
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
