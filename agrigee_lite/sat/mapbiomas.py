import ee

from agrigee_lite.ee_utils import ee_get_number_of_pixels, ee_map_valid_pixels, ee_safe_remove_borders
from agrigee_lite.sat.abstract_satellite import DataSourceSatellite


class MapBiomas(DataSourceSatellite):
    """MapBiomas Brazil Collection 10 LULC — annual land-cover map from 1985 to 2023, 30 m resolution.

    For each year in the requested date range, ``compute()`` returns one row
    with two statistics aggregated over the geometry:

    - ``class`` — the modal (most frequent) land-use/cover class code (int).
    - ``percent`` — fraction of pixels that agree with the modal class (0–1).

    This gives an annual time series of dominant land cover and its
    classification confidence for each field or polygon.

    The ``classes`` attribute maps integer codes to ``{"label", "color"}``
    dicts (e.g. ``39 → {"label": "Soybean", ...}``).

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.  Helps remove
        classification noise near geometry edges.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).

    Notes
    -----
    Coverage is Brazil only.  Classification uses Landsat imagery (TM,
    ETM+, OLI) processed with a Random Forest + temporal filtering pipeline.
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
            ("", "10_class"),
            ("", "11_percent"),
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

        subsampling_max_pixels_: float = subsampling_max_pixels if subsampling_max_pixels is not None else 1e8

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
                maxPixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels_, self.pixelSize),
                bestEffort=True,
            )
            clazz = ee.Number(mode_dict.get(year_str)).round()

            percent = (
                img.eq(clazz)
                .reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels_, self.pixelSize),
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

            stats = ee.Feature(stats.set("99_validPixelsCount", mb_image.get("ZZ_USER_VALID_PIXELS")))

            return stats

        features = years.map(_feat_for_year)
        return ee.FeatureCollection(features)

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class MapBiomasPastureVigor(DataSourceSatellite):
    """MapBiomas Brazil Collection 10 pasture vigor — annual pasture condition from 2000 to 2024, 30 m resolution.

    The MapBiomas Pasture module classifies every pasture pixel by its
    vegetative vigour trend into three condition classes: low, medium and
    high vigour.  Low vigour marks pastures with reduced forage production
    and evidence of severe (potentially biological) degradation.

    For each year in the requested date range, ``compute()`` returns one row
    with the fraction of pasture pixels in each class:

    - ``lowVigor`` — fraction of pasture pixels with low vigour (0–1).
    - ``mediumVigor`` — fraction with medium vigour (0–1).
    - ``highVigor`` — fraction with high vigour (0–1).

    The three fractions sum to 1 and are computed over the pasture pixels of
    that year only, so the dominant condition class is the ``argmax`` of the
    three.  Years in which the geometry holds no pasture pixel at all are
    omitted from the result instead of being reported as zeros, and
    ``validPixelsCount`` carries the pasture pixel count the fractions were
    computed over.

    Years outside the product's coverage (2000–2024) are clipped away, so a
    wider date range simply yields the available years.

    The ``classes`` attribute maps the source raster codes (1, 2, 3) to
    ``{"label", "color"}`` dicts, for building legends.  The colours are an
    AgriGEE.lite red–yellow–green ramp, not an official MapBiomas palette.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.  Helps remove
        classification noise near geometry edges.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).

    Notes
    -----
    Coverage is Brazil only, and only where MapBiomas maps pasture — pixels
    that were cropland, forest or anything else in a given year are masked
    out of that year's band.  This makes the product a useful external
    second opinion on a pasture degradation diagnosis, and a cheap check on
    whether an area was pasture in the first place.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__()
        self.imageAsset: str = (
            "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_pasture_vigor_v3"
        )
        self.pixelSize: int = 30
        self.firstYear: int = 2000
        self.lastYear: int = 2024
        self.startDate = "2000-01-01"
        self.endDate = "2024-12-31"
        self.shortName = "mapbiomaspasturevigor"
        self.selectedBands = [
            ("", "10_lowVigor"),
            ("", "11_mediumVigor"),
            ("", "12_highVigor"),
        ]

        self.classes = {
            1: {"label": "Low Vigor", "color": "#d73027"},
            2: {"label": "Medium Vigor", "color": "#fee08b"},
            3: {"label": "High Vigor", "color": "#1a9850"},
        }

        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode
        self.toDownloadSelectors = ["10_lowVigor", "11_mediumVigor", "12_highVigor"]

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

        vigor_image = ee.Image(self.imageAsset)

        subsampling_max_pixels_: float = subsampling_max_pixels if subsampling_max_pixels is not None else 1e8
        max_pixels = ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels_, self.pixelSize)

        indexnum = ee.Feature(ee_feature).get("0")
        years = self._years_in_coverage(ee.Feature(ee_feature))

        def _feat_for_year(year: ee.Number) -> ee.Feature:
            year_str = ee.Number(year).toInt().format()
            band_in = ee.String("classification_").cat(year_str)
            img = vigor_image.select([band_in], [year_str])

            histogram = ee.Dictionary(
                img.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=max_pixels,
                    bestEffort=True,
                ).get(year_str)
            )

            counts = [ee.Number(histogram.get(str(class_id), 0)) for class_id in self.classes]
            pasture_pixels = counts[0].add(counts[1]).add(counts[2])

            stats = {
                "00_indexnum": indexnum,
                "01_timestamp": ee.String(year_str).cat("-01-01"),
                "99_validPixelsCount": pasture_pixels,
            }
            for selector, count in zip(self.toDownloadSelectors, counts, strict=True):
                stats[selector] = ee.Algorithms.If(pasture_pixels.gt(0), count.divide(pasture_pixels), 0)

            return ee.Feature(None, stats)

        features = ee.FeatureCollection(years.map(_feat_for_year))
        return features.filter(ee.Filter.gt("99_validPixelsCount", 0))

    def _years_in_coverage(self, ee_feature: ee.Feature) -> ee.List:
        """Return the requested years clipped to the product's coverage, empty when they do not overlap."""
        first_year = ee.Number(self.firstYear)
        last_year = ee.Number(self.lastYear)

        start_year = ee.Number(ee.Date(ee_feature.get("s")).get("year")).max(first_year).min(last_year)
        end_year = ee.Number(ee.Date(ee_feature.get("e")).get("year")).min(last_year).max(first_year)

        requested_start = ee.Number(ee.Date(ee_feature.get("s")).get("year"))
        requested_end = ee.Number(ee.Date(ee_feature.get("e")).get("year"))
        overlaps = requested_start.lte(last_year).And(requested_end.gte(first_year))

        return ee.List(ee.Algorithms.If(overlaps, ee.List.sequence(start_year, end_year), ee.List([])))

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
