from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import OpticalSatellite


class Modis(OpticalSatellite):
    def __init__(self, bands: list[str] | None = None, rescale_0_1: bool = True) -> None:
        if bands is None:
            bands = ["red", "nir"]

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

        remap = {name: f"{idx}_{name}" for idx, name in enumerate(bands)}
        self.selectedBands = {remap[b]: self.availableBands[b] for b in bands if b in self.availableBands}

        self.rescale_0_1 = rescale_0_1
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

        modis_imgc = terra.merge(aqua)

        modis_imgc = ee_filter_img_collection_invalid_pixels(modis_imgc, ee_geometry, self.pixelSize, 2)

        if self.rescale_0_1:
            modis_imgc = modis_imgc.map(
                lambda img: ee.Image(img).addBands(ee.Image(img).add(100).divide(16_100), overwrite=True)
            )

        return ee.ImageCollection(modis_imgc)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
    ) -> ee.FeatureCollection:
        """Sample time series of median reflectance within *ee_feature*."""
        geom = ee_feature.geometry()
        geom = ee_safe_remove_borders(geom, self.pixelSize // 2, 190_000)

        modis = self.imageCollection(ee_feature)

        # round_int_16 is True only if reducers are None or contain exclusively 'mean' and/or 'median' and the image is not rescaled to 0-1
        allowed_reducers = {"mean", "median"}
        round_int_16 = (reducers is None or set(reducers).issubset(allowed_reducers)) and not self.rescale_0_1

        feats = modis.map(
            partial(
                ee_map_bands_and_doy,
                ee_geometry=geom,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(geom, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(["mean"] if reducers is None else reducers),
                date_types=date_types,
                round_int_16=round_int_16,
            )
        )
        return feats

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
