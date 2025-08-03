import ee

from agrigee_lite.ee_utils import (
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
    ee_map_valid_pixels,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import SingleImageSatellite


class ANADEM(SingleImageSatellite):
    def __init__(self, bands: list[str] | None = None):
        if bands is None:
            bands = ["elevation", "slope", "aspect"]

        super().__init__()

        self.imageName: str = "projects/et-brasil/assets/anadem/v1"
        self.pixelSize: int = 30
        self.shortName: str = "anadem"

        self.selectedBands: list[tuple[str, str]] = [(band, f"10_{band}") for band in bands]

        self.startDate = "1900-01-01"
        self.endDate = "2050-01-01"

    def image(self, ee_feature: ee.Feature) -> ee.Image:
        image = ee.Image(self.imageName).updateMask(ee.Image(self.imageName).neq(-9999))

        requested_bands = [b for b, _ in self.selectedBands]

        if any(b in requested_bands for b in ["slope", "aspect"]):
            terrain = ee.Terrain.products(image)
            image = image.addBands(terrain.select(["slope", "aspect"]))

        selected_band_names = [b for b, _ in self.selectedBands]
        renamed_band_names = [alias for _, alias in self.selectedBands]

        return image.select(selected_band_names, renamed_band_names)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee_safe_remove_borders(ee_geometry, self.pixelSize, 50000)
        ee_feature = ee_feature.setGeometry(ee_geometry)

        ee_img = self.image(ee_feature)
        ee_img = ee_map_valid_pixels(ee_img, ee_geometry, self.pixelSize)

        feature = ee_map_bands_and_doy(
            ee_img=ee_img,
            ee_feature=ee_feature,
            pixel_size=self.pixelSize,
            subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
            reducer=ee_get_reducers(reducers),
            single_image=True,
        )

        return ee.FeatureCollection(feature)
