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


class NASADEM(SingleImageSatellite):
    """NASADEM — reprocessed SRTM, 30 m, global 60°N–56°S (NASA / USGS, 2000).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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
        if not bands:
            raise ValueError("bands must contain at least one of: 'elevation', 'slope', 'aspect'")
        super().__init__()

        self.imageName: str = "NASA/NASADEM_HGT/001"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 30
        self.shortName: str = "nasadem"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class SRTM(SingleImageSatellite):
    """SRTM GL1 v003 — 1 arc-second (~30 m) global DEM, 60°N–56°S (NASA / USGS, 2000).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "USGS/SRTMGL1_003"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 30
        self.shortName: str = "srtm"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class CGIARSRTM(SingleImageSatellite):
    """CGIAR SRTM v4 — void-filled SRTM, 90 m, global 60°N–56°S (CGIAR-CSI, 2000).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "CGIAR/SRTM90_V4"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 90
        self.shortName: str = "cgiar_srtm"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class ASTERDEM(SingleImageSatellite):
    """ASTER GDEM v3 — stereo-optical DEM, 30 m, global 83°N–83°S (NASA / METI).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "projects/sat-io/open-datasets/ASTER/GDEM"
        self.sourceBand: str = "b1"
        self.isCollection: bool = False
        self.pixelSize: float = 30
        self.shortName: str = "aster"

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
        image = ee.Image(self.imageName)
        image = image.select(["b1"], ["elevation"])

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


class ALOSWorld3D(SingleImageSatellite):
    """ALOS AW3D30 v4.1 — PRISM-derived DSM, 30 m, global 82°N–82°S (JAXA, 2006-2011).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "JAXA/ALOS/AW3D30/V4_1"
        self.sourceBand: str = "DSM"
        self.isCollection: bool = True
        self.pixelSize: float = 30
        self.shortName: str = "alos_aw3d30"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["DSM"], ["elevation"])

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


class MERITDEM(SingleImageSatellite):
    """MERIT DEM v1.0.3 — error-removed SRTM/AW3D30, 90 m, 90°N–60°S (U. Tokyo).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "MERIT/DEM/v1_0_3"
        self.sourceBand: str = "dem"
        self.isCollection: bool = False
        self.pixelSize: float = 90
        self.shortName: str = "merit"

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
        image = ee.Image(self.imageName)
        image = image.select(["dem"], ["elevation"])

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


class GMTED2010(SingleImageSatellite):
    """GMTED2010 — multi-source global terrain, ~250 m, 84°N–56°S (USGS / NGA).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "USGS/GMTED2010_FULL"
        self.sourceBand: str = "be75"
        self.isCollection: bool = False
        self.pixelSize: float = 250
        self.shortName: str = "gmted2010"

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
        image = ee.Image(self.imageName)
        image = image.select(["be75"], ["elevation"])

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


class GTOPO30(SingleImageSatellite):
    """GTOPO30 — 30 arc-second (~1 km) global DEM (USGS EROS, 1996).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "USGS/GTOPO30"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 1000
        self.shortName: str = "gtopo30"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class ETOPO1(SingleImageSatellite):
    """ETOPO1 — 1 arc-minute (~1.8 km) global relief incl. bathymetry (NOAA NGDC, 2008).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "NOAA/NGDC/ETOPO1"
        self.sourceBand: str = "bedrock"
        self.isCollection: bool = False
        self.pixelSize: float = 1800
        self.shortName: str = "etopo1"

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
        image = ee.Image(self.imageName)
        image = image.select(["bedrock"], ["elevation"])

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


class GLOBathy(SingleImageSatellite):
    """GLOBathy — max-depth bathymetry for 1.4M global lakes/reservoirs, 30 m (sat-io, 2022).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "projects/sat-io/open-datasets/GLOBathy/GLOBathy_bathymetry"
        self.sourceBand: str = "b1"
        self.isCollection: bool = False
        self.pixelSize: float = 30
        self.shortName: str = "globathy"

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
        image = ee.Image(self.imageName)
        image = image.select(["b1"], ["elevation"])

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


class AHN2Interpolated(SingleImageSatellite):
    """Netherlands AHN2 — 0.5 m LiDAR DTM, void-interpolated (Rijkswaterstaat, 2007-2012).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "AHN/AHN2_05M_INT"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 0.5
        self.shortName: str = "ahn2_int"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class AHN2NonInterpolated(SingleImageSatellite):
    """Netherlands AHN2 — 0.5 m LiDAR DTM, voids retained (Rijkswaterstaat, 2007-2012).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "AHN/AHN2_05M_NON"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 0.5
        self.shortName: str = "ahn2_non"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class AHN2Raw(SingleImageSatellite):
    """Netherlands AHN2 — 0.5 m raw LiDAR samples (Rijkswaterstaat, 2007-2012).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "AHN/AHN2_05M_RUW"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 0.5
        self.shortName: str = "ahn2_ruw"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class AHN3(SingleImageSatellite):
    """Netherlands AHN3 — 0.5 m LiDAR DTM (Rijkswaterstaat, 2014-2019).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "AHN/AHN3"
        self.sourceBand: str = "dtm"
        self.isCollection: bool = True
        self.pixelSize: float = 0.5
        self.shortName: str = "ahn3"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["dtm"], ["elevation"])

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


class AHN4(SingleImageSatellite):
    """Netherlands AHN4 — 0.5 m LiDAR DTM (Rijkswaterstaat, 2020-2022).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "AHN/AHN4"
        self.sourceBand: str = "dtm"
        self.isCollection: bool = True
        self.pixelSize: float = 0.5
        self.shortName: str = "ahn4"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["dtm"], ["elevation"])

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


class USGS3DEP1m(SingleImageSatellite):
    """USGS 3DEP — 1 m LiDAR DTM, partial USA coverage (USGS).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "USGS/3DEP/1m"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = True
        self.pixelSize: float = 1
        self.shortName: str = "usgs_3dep_1m"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["elevation"], ["elevation"])

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


class NEONDEM(SingleImageSatellite):
    """NEON DEM — 1 m LiDAR DTM at NEON field sites, USA (NSF NEON).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "projects/neon-prod-earthengine/assets/DEM/001"
        self.sourceBand: str = "DTM"
        self.isCollection: bool = True
        self.pixelSize: float = 1
        self.shortName: str = "neon"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["DTM"], ["elevation"])

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


class England1mTerrain(SingleImageSatellite):
    """England 1 m composite DTM, 99% coverage (Environment Agency, 2000-2022).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "UK/EA/ENGLAND_1M_TERRAIN/2022"
        self.sourceBand: str = "dtm"
        self.isCollection: bool = False
        self.pixelSize: float = 1
        self.shortName: str = "england_1m"

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
        image = ee.Image(self.imageName)
        image = image.select(["dtm"], ["elevation"])

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


class FranceRGEAlti(SingleImageSatellite):
    """France RGE ALTI 1 m national DTM, metropolitan France (IGN, 2009-2021).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "IGN/RGE_ALTI/1M/2_0"
        self.sourceBand: str = "MNT"
        self.isCollection: bool = True
        self.pixelSize: float = 1
        self.shortName: str = "france_rge_alti"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["MNT"], ["elevation"])

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


class ArcticDEMMosaic(SingleImageSatellite):
    """ArcticDEM Mosaic v4.1 — 2 m stereo DEM, Arctic 50–90°N (PGC / NSF, 2012-2020).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "UMN/PGC/ArcticDEM/V4/2m_mosaic"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 2
        self.shortName: str = "arcticdem_mosaic"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class ArcticDEMStrips(SingleImageSatellite):
    """ArcticDEM Strips v3 — 2 m time-stamped stereo DEMs, Arctic (PGC / NSF, 2009-2017).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "UMN/PGC/ArcticDEM/V3/2m"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = True
        self.pixelSize: float = 2
        self.shortName: str = "arcticdem_strips"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["elevation"], ["elevation"])

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


class REMAStrips2m(SingleImageSatellite):
    """REMA Strips v1 — 2 m time-stamped stereo DEMs, Antarctica (PGC / NSF, 2009-2018).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "UMN/PGC/REMA/V1/2m"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = True
        self.pixelSize: float = 2
        self.shortName: str = "rema_strips_2m"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["elevation"], ["elevation"])

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


class REMAMosaic(SingleImageSatellite):
    """REMA Mosaic v1.1 — 8 m stereo DEM, Antarctica (PGC / NSF, 2009-2018).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "UMN/PGC/REMA/V1_1/8m"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 8
        self.shortName: str = "rema_mosaic"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class REMAStrips8m(SingleImageSatellite):
    """REMA Strips v1 — 8 m time-stamped stereo DEMs, Antarctica (PGC / NSF, 2009-2018).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "UMN/PGC/REMA/V1/8m"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = True
        self.pixelSize: float = 8
        self.shortName: str = "rema_strips_8m"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["elevation"], ["elevation"])

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


class Australia5mDEM(SingleImageSatellite):
    """Australia 5 m DEM — LiDAR/photogrammetry, expanding coverage (Geoscience Australia).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "AU/GA/AUSTRALIA_5M_DEM"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = True
        self.pixelSize: float = 5
        self.shortName: str = "australia_5m"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["elevation"], ["elevation"])

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


class USGS3DEP10m(SingleImageSatellite):
    """USGS 3DEP — 10 m DEM, contiguous USA + Alaska + Hawaii (USGS).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "USGS/3DEP/10m_collection"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = True
        self.pixelSize: float = 10
        self.shortName: str = "usgs_3dep_10m"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["elevation"], ["elevation"])

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


class CanadaCDEM(SingleImageSatellite):
    """Canadian DEM (CDEM) — ~23 m multi-source mosaic (Natural Resources Canada, 1945-2011).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "NRCan/CDEM"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = True
        self.pixelSize: float = 23
        self.shortName: str = "cdem"

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
        image = ee.ImageCollection(self.imageName).mosaic()
        image = image.select(["elevation"], ["elevation"])

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


class GreenlandGIMP(SingleImageSatellite):
    """Greenland GIMP DEM — 30 m ASTER/SPOT-5 DEM (Ohio State / NASA, 2003-2009).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "OSU/GIMP/DEM"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 30
        self.shortName: str = "gimp"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class AustraliaDEMS(SingleImageSatellite):
    """Australia DEM-S — ~30 m smoothed, hydrologically conditioned DEM (Geoscience Australia).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "AU/GA/DEM_1SEC/v10/DEM-S"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 30
        self.shortName: str = "australia_dem_s"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class AustraliaDEMH(SingleImageSatellite):
    """Australia DEM-H — ~30 m hydrologically enforced DEM (Geoscience Australia).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "AU/GA/DEM_1SEC/v10/DEM-H"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 30
        self.shortName: str = "australia_dem_h"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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


class CryoSat2Antarctica(SingleImageSatellite):
    """CryoSat-2 Antarctica DEM — 1 km radar-altimetry DEM (ESA / CPOM, 2010-2016).

    Same per-geometry terrain statistics as ``ANADEM``/``CopernicusDEM``:
    ``elevation_mean``, six ``slope_*`` class fractions, and eight
    ``cardinal_*`` aspect fractions.

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

        self.imageName: str = "CPOM/CryoSat2/ANTARCTICA_DEM"
        self.sourceBand: str = "elevation"
        self.isCollection: bool = False
        self.pixelSize: float = 1000
        self.shortName: str = "cryosat2_antarctica"

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
        image = ee.Image(self.imageName)
        image = image.select(["elevation"], ["elevation"])

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

