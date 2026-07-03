import ee

from agrigee_lite.ee_utils import (
    ee_map_valid_pixels,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import SingleImageSatellite


class WRBSoilClasses(SingleImageSatellite):
    """WRB Soil Classes — global soil classification map, 250 m resolution (single static image).

    Based on SoilGrids 2016, this product maps 30 World Reference Base (WRB)
    soil reference groups worldwide.  ``compute()`` returns the fraction of
    pixels belonging to each soil class within the geometry — one column per
    class named ``soil_<label>``.

    No parameters are needed at construction time.

    Notes
    -----
    The ``classes`` attribute maps integer class IDs to ``{"label", "color"}``
    dicts, useful for building legends when visualising results.
    """

    def __init__(self):
        super().__init__()
        self.imageName = "projects/ee-pintodasilvamateus/assets/agrigee_lite/wrb_soil_classes_2016"
        self.pixelSize = 250
        self.shortName = "wrb_soil_classes"
        self.startDate = "1900-01-01"
        self.endDate = "2050-01-01"

        self.classes = {
            0: {"label": "Acrisols", "color": "#f7991d"},
            1: {"label": "Albeluvisols", "color": "#9b9d57"},
            2: {"label": "Alisols", "color": "#faf7c0"},
            3: {"label": "Andosols", "color": "#ed3a33"},
            4: {"label": "Arenosols", "color": "#f7d8ac"},
            5: {"label": "Calcisols", "color": "#ffee00"},
            6: {"label": "Cambisols", "color": "#fecd67"},
            7: {"label": "Chernozems", "color": "#e2c837"},
            8: {"label": "Cryosols", "color": "#756a92"},
            9: {"label": "Durisols", "color": "#efe6bf"},
            10: {"label": "Ferralsols", "color": "#f6872d"},
            11: {"label": "Fluvisols", "color": "#01b0ef"},
            12: {"label": "Gleysols", "color": "#9291b9"},
            13: {"label": "Gypsisols", "color": "#fbf6a5"},
            14: {"label": "Histosols", "color": "#8b898a"},
            15: {"label": "Kastanozems", "color": "#c99580"},
            16: {"label": "Leptosols", "color": "#d5d6d8"},
            17: {"label": "Lixisols", "color": "#f9bdbf"},
            18: {"label": "Luvisols", "color": "#f48385"},
            19: {"label": "Nitisols", "color": "#f7a082"},
            20: {"label": "Phaeozems", "color": "#ba6850"},
            21: {"label": "Planosols", "color": "#f59354"},
            22: {"label": "Plinthosols", "color": "#6f0e41"},
            23: {"label": "Podzols", "color": "#0daf63"},
            24: {"label": "Regosols", "color": "#ffe2ae"},
            25: {"label": "Solonchaks", "color": "#ed3994"},
            26: {"label": "Solonetz", "color": "#f4cde2"},
            27: {"label": "Stagnosols", "color": "#40c1eb"},
            28: {"label": "Umbrisols", "color": "#618f82"},
            29: {"label": "Vertisols", "color": "#9e567c"},
        }

    def image(self, ee_feature: ee.Feature) -> ee.Image:
        return ee.Image(self.imageName).select("b1").rename("soil_class")

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: set[str] | None = None,
    ) -> ee.FeatureCollection:
        geometry = ee_safe_remove_borders(ee_feature.geometry(), self.pixelSize, 50000)
        ee_feature = ee_feature.setGeometry(geometry)

        image = self.image(ee_feature)
        image = ee_map_valid_pixels(image, geometry, self.pixelSize)

        soil = image.select("soil_class")

        total_pixels = (
            ee.Image(1)
            .updateMask(soil.mask())
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=geometry,
                scale=self.pixelSize,
                maxPixels=int(subsampling_max_pixels),
                bestEffort=True,
            )
            .getNumber("constant")
        )

        stats = {"00_indexnum": ee_feature.get("0")}

        for i, (class_id, class_info) in enumerate(self.classes.items()):
            class_mask = soil.eq(int(class_id))

            class_count = (
                ee.Image(1)
                .updateMask(class_mask)
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=geometry,
                    scale=self.pixelSize,
                    maxPixels=int(subsampling_max_pixels),
                    bestEffort=True,
                )
                .getNumber("constant")
            )

            percentage = ee.Algorithms.If(total_pixels.neq(0), ee.Number(class_count).divide(total_pixels), 0)

            key = f"{40 + i:02d}_soil_{class_info['label'].lower()}"
            stats[key] = percentage

        return ee.FeatureCollection([ee.Feature(None, stats)])


class PolarisSoilTexture(SingleImageSatellite):
    """POLARIS soil texture — clay/sand/silt content, 0-5 cm depth, 30 m resolution (CONUS only).

    Hosted by sat-io, POLARIS provides probabilistic remapping of SSURGO/
    STATSGO soil texture at 30 m over the contiguous United States.
    ``compute()`` returns the mean percent content of the requested texture
    bands plus, optionally, the fraction of pixels in each of the 12 USDA
    soil-texture-triangle classes (``usda_soil_class``).

    Parameters
    ----------
    bands : list of str, optional
        Subset of ``["clay", "sand", "silt", "usda_soil_class"]``.  Defaults
        to all four.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).

    Notes
    -----
    The ``classes`` attribute maps the 12 USDA soil-texture-triangle class
    codes to their labels.
    """

    def __init__(
        self,
        bands: list[str] | None = None,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ):
        allowed_bands = {"clay", "sand", "silt", "usda_soil_class"}

        if bands is None:
            bands = ["clay", "sand", "silt", "usda_soil_class"]

        if not bands:
            raise ValueError(f"bands must contain at least one of {sorted(allowed_bands)}")

        invalid = [b for b in bands if b not in allowed_bands]
        if invalid:
            raise ValueError(
                f"Unknown band(s) for PolarisSoilTexture: {invalid}. Valid bands are {sorted(allowed_bands)}"
            )

        super().__init__
        self.imageNames: dict[str, str] = {
            "clay": "projects/sat-io/open-datasets/polaris/clay_mean/clay_0_5",
            "sand": "projects/sat-io/open-datasets/polaris/sand_mean/sand_0_5",
            "silt": "projects/sat-io/open-datasets/polaris/silt_mean/silt_0_5",
        }
        self.pixelSize: int = 30
        self.shortName: str = "polaris_soil_texture"

        self.selectedBands: list[tuple[str, str]] = [(band, band) for band in bands]

        self.startDate = "1900-01-01"
        self.endDate = "2050-01-01"
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.classes = {
            1: "sand",
            2: "loamy_sand",
            3: "sandy_loam",
            4: "loam",
            5: "silt_loam",
            6: "silt",
            7: "sandy_clay_loam",
            8: "clay_loam",
            9: "silty_clay_loam",
            10: "sandy_clay",
            11: "silty_clay",
            12: "clay",
        }

        self.toDownloadSelectors = self._build_to_download_selectors()

    def _build_to_download_selectors(self) -> list[str]:
        selectors = []

        band_aliases = [alias for _, alias in self.selectedBands]

        if "clay" in band_aliases:
            selectors += ["40_clay_mean"]

        if "sand" in band_aliases:
            selectors += ["41_sand_mean"]

        if "silt" in band_aliases:
            selectors += ["42_silt_mean"]

        if "usda_soil_class" in band_aliases:
            selectors += [f"{43 + i:02d}_usda_{label}" for i, label in enumerate(self.classes.values())]

        return selectors

    def image(self, ee_feature: ee.Feature) -> ee.Image:
        clay = ee.Image(self.imageNames["clay"])
        sand = ee.Image(self.imageNames["sand"])
        silt = ee.Image(self.imageNames["silt"])
        composite = ee.Image.cat([clay, sand, silt]).rename(["clay", "sand", "silt"])

        requested_bands = [b for b, _ in self.selectedBands]

        if "usda_soil_class" in requested_bands:
            usda_soil_class = composite.expression(
                "(b('silt') + 1.5 * b('clay') < 15) ? 1 : "
                "(b('silt') + 1.5 * b('clay') >= 15 && b('silt') + 2 * b('clay') < 30) ? 2 : "
                "(b('clay') >= 7 && b('clay') < 20 && b('sand') > 52 && b('silt') + 2 * b('clay') >= 30) || "
                "(b('clay') < 7 && b('silt') < 50 && b('silt') + 2 * b('clay') >= 30) ? 3 : "
                "(b('clay') >= 7 && b('clay') < 27 && b('silt') >= 28 && b('silt') < 50 && b('sand') <= 52) ? 4 : "
                "(b('silt') >= 50 && b('clay') >= 12 && b('clay') < 27) || "
                "(b('silt') >= 50 && b('silt') < 80 && b('clay') < 12) ? 5 : "
                "(b('silt') >= 80 && b('clay') < 12) ? 6 : "
                "(b('clay') >= 20 && b('clay') < 35 && b('silt') < 28 && b('sand') > 45) ? 7 : "
                "(b('clay') >= 27 && b('clay') < 40 && b('sand') > 20 && b('sand') <= 45) ? 8 : "
                "(b('clay') >= 27 && b('clay') < 40 && b('sand') <= 20) ? 9 : "
                "(b('clay') >= 35 && b('sand') > 45) ? 10 : "
                "(b('clay') >= 40 && b('silt') >= 40) ? 11 : "
                "(b('clay') >= 40 && b('sand') <= 45 && b('silt') < 40) ? 12 : 1"
            ).rename("usda_soil_class")
            composite = composite.addBands(usda_soil_class)

        selected_band_names = [b for b, _ in self.selectedBands]

        return composite.select(selected_band_names)

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

        # --- Texture percentage means ---
        for band, prefix in (("clay", "40"), ("sand", "41"), ("silt", "42")):
            if band in selected_band_names:
                mean_value = (
                    ee_img.select(band)
                    .reduceRegion(
                        reducer=ee.Reducer.mean(),
                        geometry=ee_geometry,
                        scale=self.pixelSize,
                        maxPixels=int(subsampling_max_pixels),
                        bestEffort=True,
                    )
                    .get(band)
                )
                stats_dict[f"{prefix}_{band}_mean"] = mean_value

        # --- USDA soil class breakdown ---
        if "usda_soil_class" in selected_band_names:
            usda = ee_img.select("usda_soil_class")

            valid_mask = usda.mask()
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

            for i, (class_id, label) in enumerate(self.classes.items()):
                class_mask = usda.eq(int(class_id))

                count = (
                    ee.Image(1)
                    .updateMask(class_mask)
                    .reduceRegion(
                        reducer=ee.Reducer.count(),
                        geometry=ee_geometry,
                        scale=self.pixelSize,
                        maxPixels=int(subsampling_max_pixels),
                        bestEffort=True,
                    )
                    .getNumber("constant")
                )

                percent = ee.Algorithms.If(total_pixels.neq(0), ee.Number(count).divide(total_pixels), 0)
                stats_dict[f"{43 + i:02d}_usda_{label}"] = percent

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
