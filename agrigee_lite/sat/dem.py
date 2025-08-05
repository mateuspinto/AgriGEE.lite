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
    def __init__(self, bands: list[str] | None = None, mode: str = "by_class"):
        if bands is None:
            bands = ["elevation", "slope", "aspect"]

        if mode not in ("default", "by_class"):
            raise ValueError(f"Unsupported mode '{mode}'. Use 'default' or 'by_class'.")  # noqa: TRY003

        super().__init__()

        self.imageName: str = "projects/et-brasil/assets/anadem/v1"
        self.pixelSize: int = 30
        self.shortName: str = "anadem"
        self.mode: str = mode

        self.selectedBands: list[tuple[str, str]] = [(band, f"{band}") for band in bands]

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
        if self.mode == "default":
            return self._compute_default(ee_feature, subsampling_max_pixels, reducers)
        elif self.mode == "by_class":
            return self._compute_by_class(ee_feature, subsampling_max_pixels)
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'")  # noqa: TRY003

    def _compute_default(
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

    def _compute_by_class(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee_safe_remove_borders(ee_geometry, self.pixelSize, 50000)
        ee_feature = ee_feature.setGeometry(ee_geometry)

        ee_img = self.image(ee_feature)
        ee_img = ee_map_valid_pixels(ee_img, ee_geometry, self.pixelSize)

        # Total valid pixels (for slope class %)
        valid_mask = ee_img.select("elevation").mask()
        total_pixels = (
            ee.Image(1)
            .updateMask(valid_mask)
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=ee_geometry,
                scale=self.pixelSize,
                maxPixels=subsampling_max_pixels,
                bestEffort=True,
            )
            .getNumber("constant")
        )

        # Mean elevation
        elevation_mean = (
            ee_img.select("elevation")
            .reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geometry,
                scale=self.pixelSize,
                maxPixels=subsampling_max_pixels,
                bestEffort=True,
            )
            .get("elevation")
        )

        # Slope class breakdown
        slope = ee_img.select("slope")

        slope_classes = {
            "41_slope_flat": slope.gte(0).And(slope.lt(3)),
            "42_slope_gentle": slope.gte(3).And(slope.lt(8)),
            "43_slope_undulating": slope.gte(8).And(slope.lt(20)),
            "44_slope_strong": slope.gte(20).And(slope.lt(45)),
            "45_slope_mountainous": slope.gte(45).And(slope.lte(75)),
            "46_slope_steep": slope.gt(75),
        }

        class_percentages = {}
        for class_name, mask in slope_classes.items():
            count = (
                ee.Image(1)
                .updateMask(mask)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=subsampling_max_pixels,
                    bestEffort=True,
                )
                .getNumber("constant")
            )
            percent = count.divide(total_pixels)
            class_percentages[class_name] = percent

            # Aspect class breakdown
        aspect = ee_img.select("aspect")

        aspect_classes = {
            "47_cardinal_n": aspect.gte(337.5).Or(aspect.lt(22.5)),
            "48_cardinal_ne": aspect.gte(22.5).And(aspect.lt(67.5)),
            "49_cardinal_e": aspect.gte(67.5).And(aspect.lt(112.5)),
            "50_cardinal_se": aspect.gte(112.5).And(aspect.lt(157.5)),
            "51_cardinal_s": aspect.gte(157.5).And(aspect.lt(202.5)),
            "52_cardinal_sw": aspect.gte(202.5).And(aspect.lt(247.5)),
            "53_cardinal_w": aspect.gte(247.5).And(aspect.lt(292.5)),
            "54_cardinal_nw": aspect.gte(292.5).And(aspect.lt(337.5)),
            # "55_cardinal_flat": aspect.eq(0).Not(),  # we'll subtract from total later
        }

        # compute valid aspect pixels for normalization
        valid_aspect = aspect.mask()
        total_aspect_pixels = (
            ee.Image(1)
            .updateMask(valid_aspect)
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=ee_geometry,
                scale=self.pixelSize,
                maxPixels=subsampling_max_pixels,
                bestEffort=True,
            )
            .getNumber("constant")
        )

        aspect_percentages = {}
        for class_name, mask in aspect_classes.items():
            count = (
                ee.Image(1)
                .updateMask(mask)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=subsampling_max_pixels,
                    bestEffort=True,
                )
                .getNumber("constant")
            )
            percent = count.divide(total_aspect_pixels)
            aspect_percentages[class_name] = percent

        stats_dict = {
            "00_indexnum": ee_feature.get("0"),
            "40_elevation_mean": elevation_mean,
            **class_percentages,
            **aspect_percentages,
        }

        stats_feature = ee.Feature(None, stats_dict)
        return ee.FeatureCollection([stats_feature])
