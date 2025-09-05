import ee

from agrigee_lite.ee_utils import (
    ee_map_valid_pixels,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import SingleImageSatellite


class WRBSoilClasses(SingleImageSatellite):
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

    def image(self, _: ee.Feature) -> ee.Image:
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
                maxPixels=subsampling_max_pixels,
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
                    maxPixels=subsampling_max_pixels,
                    bestEffort=True,
                )
                .getNumber("constant")
            )

            percentage = ee.Algorithms.If(total_pixels.neq(0), ee.Number(class_count).divide(total_pixels), 0)

            key = f"{40 + i:02d}_soil_{class_info['label'].lower()}"
            stats[key] = percentage

        return ee.FeatureCollection([ee.Feature(None, stats)])
