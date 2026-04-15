import ee

from agrigee_lite.ee_utils import (
    ee_map_valid_pixels,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import SingleImageSatellite


class ANADEM(SingleImageSatellite):
    """ANADEM — Brazilian territory DEM, 30 m resolution (single static image).

    Topographic product covering Brazil, derived from SRTM and auxiliary DEMs
    by FURGS / ANA.  Unlike time-series satellites, this returns one row per
    geometry with terrain statistics.

    ``compute()`` produces:
    - ``elevation_mean`` — mean elevation (m above sea level).
    - ``slope_*`` — fraction of pixels in six slope classes (flat ≤3°,
      gentle 3–8°, undulating 8–20°, strong 20–45°, mountainous 45–75°,
      steep >75°).
    - ``cardinal_*`` — fraction of pixels in each of the eight compass
      aspect directions.

    Parameters
    ----------
    bands : list of str, optional
        Subset of ``["elevation", "slope", "aspect"]``.  Defaults to all three.
        Omitting a band also omits its derived statistics from the output.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).
    """

    def __init__(
        self,
        bands: list[str] | None = None,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ):
        if bands is None:
            bands = ["elevation", "slope", "aspect"]

        super().__init__()

        self.imageName: str = "projects/et-brasil/assets/anadem/v1"
        self.pixelSize: int = 30
        self.shortName: str = "anadem"

        self.selectedBands: list[tuple[str, str]] = [(band, f"{band}") for band in bands]

        self.startDate = "1900-01-01"
        self.endDate = "2050-01-01"
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.toDownloadSelectors = self._build_to_download_selectors()

    def _build_to_download_selectors(self) -> list[str]:
        selectors = []

        band_aliases = [alias for _, alias in self.selectedBands]

        if "elevation" in band_aliases:
            selectors += ["40_elevation_mean"]

        if "slope" in band_aliases:
            selectors += [
                "41_slope_flat",
                "42_slope_gentle",
                "43_slope_undulating",
                "44_slope_strong",
                "45_slope_mountainous",
                "46_slope_steep",
            ]

        if "aspect" in band_aliases:
            selectors += [
                "47_cardinal_n",
                "48_cardinal_ne",
                "49_cardinal_e",
                "50_cardinal_se",
                "51_cardinal_s",
                "52_cardinal_sw",
                "53_cardinal_w",
                "54_cardinal_nw",
            ]

        return selectors

    def image(self, ee_feature: ee.Feature) -> ee.Image:
        image = ee.Image(self.imageName).updateMask(ee.Image(self.imageName).neq(ee.Number(-9999)))

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
        reducers: set[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()

        if self.borderPixelsToErode != 0:
            ee_geometry = ee_safe_remove_borders(
                ee_geometry, round(self.borderPixelsToErode * self.pixelSize), self.minAreaToKeepBorder
            )
            ee_feature = ee_feature.setGeometry(ee_geometry)

        ee_img = self.image(ee_feature)
        ee_img = ee_map_valid_pixels(ee_img, ee_geometry, self.pixelSize)

        selected_band_names = [alias for _, alias in self.selectedBands]

        stats_dict = {
            "00_indexnum": ee_feature.get("0"),
        }

        # --- Elevation mean ---
        if "elevation" in selected_band_names:
            elevation_mean = (
                ee_img.select("elevation")
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=int(subsampling_max_pixels),
                    bestEffort=True,
                )
                .get("elevation")
            )
            stats_dict["40_elevation_mean"] = elevation_mean

        # --- Slope class breakdown ---
        if "slope" in selected_band_names:
            slope = ee_img.select("slope")

            slope_classes = {
                "41_slope_flat": slope.gte(0).And(slope.lt(3)),
                "42_slope_gentle": slope.gte(3).And(slope.lt(8)),
                "43_slope_undulating": slope.gte(8).And(slope.lt(20)),
                "44_slope_strong": slope.gte(20).And(slope.lt(45)),
                "45_slope_mountainous": slope.gte(45).And(slope.lte(75)),
                "46_slope_steep": slope.gt(75),
            }

            valid_mask = ee_img.select("slope").mask()
            total_pixels = (
                ee.Image(1)
                .updateMask(valid_mask)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=int(subsampling_max_pixels),
                    bestEffort=True,
                )
                .getNumber("constant")
            )

            for class_name, mask in slope_classes.items():
                count = (
                    ee.Image(1)
                    .updateMask(mask)
                    .reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=ee_geometry,
                        scale=self.pixelSize,
                        maxPixels=int(subsampling_max_pixels),
                        bestEffort=True,
                    )
                    .getNumber("constant")
                )
                percent = count.divide(total_pixels)
                stats_dict[class_name] = percent

        # --- Aspect class breakdown ---
        if "aspect" in selected_band_names:
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
            }

            valid_aspect = aspect.mask()
            total_aspect_pixels = (
                ee.Image(1)
                .updateMask(valid_aspect)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=int(subsampling_max_pixels),
                    bestEffort=True,
                )
                .getNumber("constant")
            )

            for class_name, mask in aspect_classes.items():
                count = (
                    ee.Image(1)
                    .updateMask(mask)
                    .reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=ee_geometry,
                        scale=self.pixelSize,
                        maxPixels=int(subsampling_max_pixels),
                        bestEffort=True,
                    )
                    .getNumber("constant")
                )
                percent = count.divide(total_aspect_pixels)
                stats_dict[class_name] = percent

        # --- ValidPixelCount ---
        valid_pixel_count = (
            ee_img.select(selected_band_names[0])
            .mask()
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=ee_geometry,
                scale=self.pixelSize,
                maxPixels=subsampling_max_pixels,
                bestEffort=True,
            )
            .getNumber(selected_band_names[0])
        )
        stats_dict["99_validPixelsCount"] = valid_pixel_count

        stats_feature = ee.Feature(None, stats_dict)
        return ee.FeatureCollection([stats_feature])


class CopernicusDEM(SingleImageSatellite):
    """Copernicus DEM GLO30 — global DEM, 30 m resolution (single static image).

    Global Digital Elevation Model from ESA's Copernicus programme.  Same
    output schema as ``ANADEM`` but with worldwide coverage.

    ``compute()`` produces:
    - ``elevation_mean`` — mean elevation (m above sea level).
    - ``slope_*`` — fraction of pixels in six slope classes (flat ≤3°,
      gentle 3–8°, undulating 8–20°, strong 20–45°, mountainous 45–75°,
      steep >75°).
    - ``cardinal_*`` — fraction of pixels in each of the eight compass
      aspect directions.

    Parameters
    ----------
    bands : list of str, optional
        Subset of ``["elevation", "slope", "aspect"]``.  Defaults to all three.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).
    """

    def __init__(
        self,
        bands: list[str] | None = None,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ):
        if bands is None:
            bands = ["elevation", "slope", "aspect"]

        super().__init__()

        self.imageName: str = "COPERNICUS/DEM/GLO30"
        self.pixelSize: int = 30
        self.shortName: str = "copdem"

        self.selectedBands: list[tuple[str, str]] = [(band, f"{band}") for band in bands]

        self.startDate = "1900-01-01"
        self.endDate = "2050-01-01"
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.toDownloadSelectors = self._build_to_download_selectors()

    def _build_to_download_selectors(self) -> list[str]:
        selectors = []

        band_aliases = [alias for _, alias in self.selectedBands]

        if "elevation" in band_aliases:
            selectors += ["40_elevation_mean"]

        if "slope" in band_aliases:
            selectors += [
                "41_slope_flat",
                "42_slope_gentle",
                "43_slope_undulating",
                "44_slope_strong",
                "45_slope_mountainous",
                "46_slope_steep",
            ]

        if "aspect" in band_aliases:
            selectors += [
                "47_cardinal_n",
                "48_cardinal_ne",
                "49_cardinal_e",
                "50_cardinal_se",
                "51_cardinal_s",
                "52_cardinal_sw",
                "53_cardinal_w",
                "54_cardinal_nw",
            ]

        return selectors

    def image(self, ee_feature: ee.Feature) -> ee.Image:
        image = ee.Image(self.imageName).updateMask(ee.Image(self.imageName).neq(ee.Number(-32768)))

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
        reducers: set[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()

        if self.borderPixelsToErode != 0:
            ee_geometry = ee_safe_remove_borders(
                ee_geometry, round(self.borderPixelsToErode * self.pixelSize), self.minAreaToKeepBorder
            )
            ee_feature = ee_feature.setGeometry(ee_geometry)

        ee_img = self.image(ee_feature)
        ee_img = ee_map_valid_pixels(ee_img, ee_geometry, self.pixelSize)

        selected_band_names = [alias for _, alias in self.selectedBands]

        stats_dict = {
            "00_indexnum": ee_feature.get("0"),
        }

        # --- Elevation mean ---
        if "elevation" in selected_band_names:
            elevation_mean = (
                ee_img.select("elevation")
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=int(subsampling_max_pixels),
                    bestEffort=True,
                )
                .get("elevation")
            )
            stats_dict["40_elevation_mean"] = elevation_mean

        # --- Slope class breakdown ---
        if "slope" in selected_band_names:
            slope = ee_img.select("slope")

            slope_classes = {
                "41_slope_flat": slope.gte(0).And(slope.lt(3)),
                "42_slope_gentle": slope.gte(3).And(slope.lt(8)),
                "43_slope_undulating": slope.gte(8).And(slope.lt(20)),
                "44_slope_strong": slope.gte(20).And(slope.lt(45)),
                "45_slope_mountainous": slope.gte(45).And(slope.lte(75)),
                "46_slope_steep": slope.gt(75),
            }

            valid_mask = ee_img.select("slope").mask()
            total_pixels = (
                ee.Image(1)
                .updateMask(valid_mask)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=int(subsampling_max_pixels),
                    bestEffort=True,
                )
                .getNumber("constant")
            )

            for class_name, mask in slope_classes.items():
                count = (
                    ee.Image(1)
                    .updateMask(mask)
                    .reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=ee_geometry,
                        scale=self.pixelSize,
                        maxPixels=int(subsampling_max_pixels),
                        bestEffort=True,
                    )
                    .getNumber("constant")
                )
                percent = count.divide(total_pixels)
                stats_dict[class_name] = percent

        # --- Aspect class breakdown ---
        if "aspect" in selected_band_names:
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
            }

            valid_aspect = aspect.mask()
            total_aspect_pixels = (
                ee.Image(1)
                .updateMask(valid_aspect)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=int(subsampling_max_pixels),
                    bestEffort=True,
                )
                .getNumber("constant")
            )

            for class_name, mask in aspect_classes.items():
                count = (
                    ee.Image(1)
                    .updateMask(mask)
                    .reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=ee_geometry,
                        scale=self.pixelSize,
                        maxPixels=int(subsampling_max_pixels),
                        bestEffort=True,
                    )
                    .getNumber("constant")
                )
                percent = count.divide(total_aspect_pixels)
                stats_dict[class_name] = percent

        # --- ValidPixelCount ---
        valid_pixel_count = (
            ee_img.select(selected_band_names[0])
            .mask()
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=ee_geometry,
                scale=self.pixelSize,
                maxPixels=subsampling_max_pixels,
                bestEffort=True,
            )
            .getNumber(selected_band_names[0])
        )
        stats_dict["99_validPixelsCount"] = valid_pixel_count

        stats_feature = ee.Feature(None, stats_dict)
        return ee.FeatureCollection([stats_feature])
