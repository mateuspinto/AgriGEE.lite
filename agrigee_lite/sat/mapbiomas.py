import ee

from agrigee_lite.ee_utils import ee_safe_remove_borders
from agrigee_lite.sat.abstract_satellite import DataSourceSatellite


class MapBiomas(DataSourceSatellite):
    def __init__(self) -> None:
        super().__init__()
        self.imageAsset: str = (
            "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1"
        )
        self.pixelSize: int = 30
        self.startDate = "1985-02-24"
        self.endDate = "2023-12-31"
        self.shortName = "mapbiomasmajclass"

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float | None = None,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee_safe_remove_borders(ee_geometry, self.pixelSize, 50000)

        mb_image = ee.Image(self.imageAsset)

        ee_start = ee.Feature(ee_feature).get("s")
        ee_end = ee.Feature(ee_feature).get("e")
        start_year = ee.Date(ee_start).get("year")
        end_year = ee.Date(ee_end).get("year")
        indexnum = ee.Feature(ee_feature).get("0")

        years = ee.List.sequence(start_year, end_year)

        def _feat_for_year(year: ee.Number) -> ee.Feature:
            year_num = ee.Number(year).toInt()
            year_str = year_num.format()
            band_in = ee.String("classification_").cat(year_str)
            img = mb_image.select([band_in], [year_str])

            mode_dict = img.reduceRegion(
                reducer=ee.Reducer.mode(),
                geometry=ee_geometry,
                scale=self.pixelSize,
                maxPixels=1e13,
                bestEffort=True,
            )
            clazz = ee.Number(mode_dict.get(year_str)).round()

            percent = (
                img.eq(clazz)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=1e13,
                    bestEffort=True,
                )
                .get(year_str)
            )
            return ee.Feature(
                None,
                {
                    "00_indexnum": indexnum,
                    "02_year": year,
                    "10_class": clazz,
                    "11_percent": percent,
                },
            )

        features = years.map(_feat_for_year)
        return ee.FeatureCollection(features)

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
