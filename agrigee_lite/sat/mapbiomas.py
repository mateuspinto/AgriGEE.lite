import ee

from agrigee_lite.ee_utils import ee_get_number_of_pixels, ee_map_valid_pixels, ee_safe_remove_borders
from agrigee_lite.sat.abstract_satellite import DataSourceSatellite


class MapBiomas(DataSourceSatellite):
    """
    Satellite abstraction for MapBiomas Brazil Collection 10 Land Use and Land Cover (LULC) data.

    This class wraps the official MapBiomas Collection 10 LULC classification product for Brazil.
    The dataset provides annual land use and land cover classifications from 1985 to 2023 at 30-meter resolution,
    with majority class (`10_class`) and percent agreement (`11_percent`) bands.

    It is suitable for long-term land cover trend analysis, ecosystem monitoring, and environmental assessments.

    Parameters
    ----------
    border_pixels_to_erode : float, default=1
        Number of border pixels (in pixels) to erode from the input geometry before analysis.
        Helps remove classification noise from edges. Use 0 to disable.
    min_area_to_keep_border : int, default=50000
        Minimum area in square meters to retain the eroded region.
        Used to avoid discarding small geometries entirely.

    Bands
    -----
    +-------------+------------------------+-------------------------------------------------------------+
    | Band Name   | Type                   | Description                                                 |
    +-------------+------------------------+-------------------------------------------------------------+
    | 10_class    | Categorical (int)      | Most frequent land use/cover class for the pixel/year       |
    | 11_percent  | Float (0-1)            | Proportion of classification votes for the majority class   |
    +-------------+------------------------+-------------------------------------------------------------+

    Classes
    -------
    Each integer value in the `10_class` band corresponds to a LULC class defined by MapBiomas.
    Refer to `self.classes` for full label and color mapping. Examples include:
    - 3: Forest Formation
    - 14: Farming
    - 24: Urban Area
    - 26: Water
    - 39: Soybean
    - 46: Coffee

    Processing Overview
    -------------------
    1. The MapBiomas classification image (`mapbiomas_brazil_collection10_coverage_v2`) is loaded.
    2. For each year between the start and end date of the input feature:
    - The modal (most frequent) class is computed (`10_class`)
    - Its pixel agreement (% of pixels matching that class) is calculated (`11_percent`)
    3. Optionally, the geometry is eroded to avoid edge noise.
    4. Final features are returned as an annual time series of LULC summaries.

    Dataset Information
    -------------------
    +-------------------------+------------------------------------------------------+
    | Field                   | Value                                                |
    +-------------------------+------------------------------------------------------+
    | Dataset                 | MapBiomas Brazil Collection 10                       |
    | Temporal Coverage       | 1985 - 2023                                          |
    | Spatial Resolution      | 30 meters                                            |
    | Projection              | EPSG: 4674 (SIRGAS 2000)                             |
    | Source Imagery          | Landsat (TM, ETM+, OLI)                              |
    | Classification Method   | Random Forest + Temporal Filtering                   |
    +-------------------------+------------------------------------------------------+

    Notes
    -----
    - Official MapBiomas dataset (Earth Engine):
    https://developers.google.com/earth-engine/datasets/catalog/projects_mapbiomas-public_assets_brazil_lulc_collection10_mapbiomas_brazil_collection10_coverage_v2

    - ATBD (Algorithm Theoretical Basis Document) Collection 10:
    https://mapbiomas.org/downloads?cama_set_language=en

    - Only the majority class (`classification_YEAR`) is used here — secondary confidence or transitions are not included.

    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__()
        self.imageAsset: str = (
            "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_coverage_v2"
        )
        self.pixelSize: int = 30
        self.startDate = "1985-01-01"
        self.endDate = "2024-01-01"
        self.shortName = "mapbiomasmajclass"
        self.selectedBands = [
            (None, "10_class"),
            (None, "11_percent"),
        ]

        self.classes = {
            1: {"label": "Forest", "color": "#1f8d49"},
            3: {"label": "Forest Formation", "color": "#1f8d49"},
            4: {"label": "Savanna Formation", "color": "#7dc975"},
            5: {"label": "Mangrove", "color": "#04381d"},
            6: {"label": "Floodable Forest", "color": "#007785"},
            9: {"label": "Forest Plantation", "color": "#7a5900"},
            10: {"label": "Herbaceous and Shrubby Vegetation", "color": "#d6bc74"},
            11: {"label": "Wetland", "color": "#519799"},
            12: {"label": "Grassland", "color": "#d6bc74"},
            14: {"label": "Farming", "color": "#ffefc3"},
            15: {"label": "Pasture", "color": "#edde8e"},
            18: {"label": "Agriculture", "color": "#E974ED"},
            19: {"label": "Temporary Crop", "color": "#C27BA0"},
            20: {"label": "Sugar cane", "color": "#db7093"},
            21: {"label": "Mosaic of Uses", "color": "#ffefc3"},
            22: {"label": "Non vegetated area", "color": "#d4271e"},
            23: {"label": "Beach, Dune and Sand Spot", "color": "#ffa07a"},
            24: {"label": "Urban Area", "color": "#d4271e"},
            25: {"label": "Other non Vegetated Areas", "color": "#db4d4f"},
            26: {"label": "Water", "color": "#2532e4"},
            27: {"label": "Not Observed", "color": "#ffffff"},
            29: {"label": "Rocky Outcrop", "color": "#ffaa5f"},
            30: {"label": "Mining", "color": "#9c0027"},
            31: {"label": "Aquaculture", "color": "#091077"},
            32: {"label": "Hypersaline Tidal Flat", "color": "#fc8114"},
            33: {"label": "River, Lake and Ocean", "color": "#2532e4"},
            35: {"label": "Palm Oil", "color": "#9065d0"},
            36: {"label": "Perennial Crop", "color": "#d082de"},
            39: {"label": "Soybean", "color": "#f5b3c8"},
            40: {"label": "Rice", "color": "#c71585"},
            41: {"label": "Other Temporary Crops", "color": "#f54ca9"},
            46: {"label": "Coffee", "color": "#d68fe2"},
            47: {"label": "Citrus", "color": "#9932cc"},
            48: {"label": "Other Perennial Crops", "color": "#e6ccff"},
            49: {"label": "Wooded Sandbank Vegetation", "color": "#02d659"},
            50: {"label": "Herbaceous Sandbank Vegetation", "color": "#ad5100"},
            62: {"label": "Cotton (beta)", "color": "#ff69b4"},
            75: {"label": "Photovoltaic Power Plant (beta)", "color": "#c12100"},
        }

        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode
        self.toDownloadSelectors = ["10_class", "11_percent"]

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float | None = None,
        reducers: set[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()

        if self.borderPixelsToErode != 0:
            ee_geometry = ee_safe_remove_borders(
                ee_geometry, round(self.borderPixelsToErode * self.pixelSize), self.minAreaToKeepBorder
            )
            ee_feature = ee_feature.setGeometry(ee_geometry)

        mb_image = ee.Image(self.imageAsset)
        mb_image = ee_map_valid_pixels(mb_image, ee_geometry, self.pixelSize)

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
                maxPixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                bestEffort=True,
            )
            clazz = ee.Number(mode_dict.get(year_str)).round()

            percent = (
                img.eq(clazz)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                    bestEffort=True,
                )
                .get(year_str)
            )

            timestamp = ee.String(year_str).cat("-01-01")

            stats = ee.Feature(
                None,
                {
                    "00_indexnum": indexnum,
                    "01_timestamp": timestamp,
                    "10_class": clazz,
                    "11_percent": percent,
                },
            )

            stats = stats.set("99_validPixelsCount", mb_image.get("ZZ_USER_VALID_PIXELS"))

            return stats

        features = years.map(_feat_for_year)
        return ee.FeatureCollection(features)

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
