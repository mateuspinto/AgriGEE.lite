from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
)
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


class Modis(AbstractSatellite):
    def __init__(self, bands: list[str] | None = None) -> None:
        if bands is None:
            bands = ["red", "nir"]

        super().__init__()

        self.shortName = "modis"
        self.pixelSize = 250
        self.startDate = "2000-02-24"
        self.endDate = "present"

        self._terra_vis = "MODIS/061/MOD09GQ"
        self._terra_qa = "MODIS/061/MOD09GA"
        self._aqua_vis = "MODIS/061/MYD09GQ"
        self._aqua_qa = "MODIS/061/MYD09GA"

        self.availableBands = {
            "red": "sur_refl_b01",
            "nir": "sur_refl_b02",
        }

        remap = {name: f"{idx}_{name}" for idx, name in enumerate(bands)}
        self.selectedBands = {remap[b]: self.availableBands[b] for b in bands if b in self.availableBands}

        self.scaleBands = lambda img: img

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
                    list(self.selectedBands.values()),
                    list(self.selectedBands.keys()),
                )
            )

        terra = _base(self._terra_vis, self._terra_qa)
        aqua = _base(self._aqua_vis, self._aqua_qa)

        col = terra.merge(aqua)

        col = ee_filter_img_collection_invalid_pixels(col, ee_geometry, self.pixelSize, 2)

        return ee.ImageCollection(col)

    def compute(
        self,
        ee_feature: ee.Feature,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
        subsampling_max_pixels: float = 1600,
    ) -> ee.FeatureCollection:
        """Sample time series of median reflectance within *ee_feature*."""
        geom = ee_feature.geometry()

        modis = self.imageCollection(ee_feature)

        subsample = ee_get_number_of_pixels(geom, subsampling_max_pixels, self.pixelSize)

        feats = modis.map(
            partial(
                ee_map_bands_and_doy,
                ee_geometry=geom,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=subsample,
                reducer=ee_get_reducers(reducers),
                date_types=date_types,
            )
        )
        return feats

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
