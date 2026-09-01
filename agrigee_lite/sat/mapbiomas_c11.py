"""MapBiomas Brazil Collection 11 data sources.

Collection 11 covers 1985-2025 and lives beside Collection 10 rather than
replacing it: the class codes and the mapped extent differ between collections,
so a silent switch would change existing results. The Collection 10 classes stay
in ``agrigee_lite.sat.mapbiomas``.

Every Collection 11 asset is a single ``ee.Image`` whose bands are named
``classification_<year>``, which is why all of these classes share one shape:
pick the band for a year, reduce it over the geometry, emit one row per year.

Two of the products here are published without a legend. MapBiomas has not
released the code tables for mined substance or for deforestation/secondary
vegetation, so those classes report raw codes and leave ``classes`` empty rather
than inventing labels.
"""

import ee

from agrigee_lite.ee_utils import (
    ee_get_number_of_pixels,
    ee_safe_remove_borders,
    ee_years_in_coverage,
)
from agrigee_lite.sat.abstract_satellite import DataSourceSatellite, SingleImageSatellite

_COLLECTION11 = "projects/mapbiomas-public/assets/brazil/lulc/collection11"


class _AnnualClassFractions(DataSourceSatellite):
    """Shared machinery for the annual Collection 11 class products.

    Subclasses declare an asset, a coverage window and a ``{code: column}``
    mapping; this class turns those into one row per year holding the fraction
    of pixels in each class.

    The denominator is every pixel the product actually classified inside the
    geometry, which is what ``99_validPixelsCount`` reports. That distinction
    carries real meaning in these products: MapBiomas masks pixels it does not
    map at all (outside the product's extent, or not the relevant land use that
    year), so a fraction of 0 means "mapped, and none of this class" while a
    dropped row means "not mapped here". Collapsing the two would turn an
    unmapped field into a confident zero.
    """

    def __init__(
        self,
        border_pixels_to_erode: float,
        min_area_to_keep_border: int,
    ) -> None:
        super().__init__()
        self.imageAsset: str = ""
        self.firstYear: int = 0
        self.lastYear: int = 0
        self.classColumns: dict[int, str] = {}
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

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

        source_image = ee.Image(self.imageAsset)

        subsampling_max_pixels_: float = subsampling_max_pixels if subsampling_max_pixels is not None else 1e8
        max_pixels = ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels_, self.pixelSize)

        indexnum = ee.Feature(ee_feature).get("0")
        years = ee_years_in_coverage(ee.Feature(ee_feature), self.firstYear, self.lastYear)

        def _feat_for_year(year: ee.Number) -> ee.Feature:
            year_str = ee.Number(year).toInt().format()
            band_in = ee.String("classification_").cat(year_str)
            img = source_image.select([band_in], [year_str])

            histogram = ee.Dictionary(
                img.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=max_pixels,
                    bestEffort=True,
                ).get(year_str)
            )

            counts = [ee.Number(histogram.get(str(code), 0)) for code in self.classColumns]
            classified_pixels = ee.Number(0)
            for count in counts:
                classified_pixels = classified_pixels.add(count)

            stats: dict = {
                "00_indexnum": indexnum,
                "01_timestamp": ee.String(year_str).cat("-01-01"),
                "99_validPixelsCount": classified_pixels,
            }
            for column, count in zip(self.classColumns.values(), counts, strict=True):
                stats[column] = ee.Algorithms.If(
                    classified_pixels.gt(0), count.divide(classified_pixels), 0
                )

            return ee.Feature(None, stats)

        features = ee.FeatureCollection(years.map(_feat_for_year))
        return features.filter(ee.Filter.gt("99_validPixelsCount", 0))

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class _AnnualModalClass(DataSourceSatellite):
    """Shared machinery for the annual Collection 11 products with wide legends.

    Where a product has dozens of codes — the full land cover legend, or the
    hierarchical mined-substance codes — one column per class is unusable, so
    these report the modal code and how much of the geometry agrees with it.
    """

    def __init__(
        self,
        border_pixels_to_erode: float,
        min_area_to_keep_border: int,
    ) -> None:
        super().__init__()
        self.imageAsset: str = ""
        self.firstYear: int = 0
        self.lastYear: int = 0
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode
        self.selectedBands = [
            ("", "10_class"),
            ("", "11_percent"),
        ]
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

        source_image = ee.Image(self.imageAsset)

        subsampling_max_pixels_: float = subsampling_max_pixels if subsampling_max_pixels is not None else 1e8
        max_pixels = ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels_, self.pixelSize)

        indexnum = ee.Feature(ee_feature).get("0")
        years = ee_years_in_coverage(ee.Feature(ee_feature), self.firstYear, self.lastYear)

        def _feat_for_year(year: ee.Number) -> ee.Feature:
            year_str = ee.Number(year).toInt().format()
            band_in = ee.String("classification_").cat(year_str)
            img = source_image.select([band_in], [year_str])

            histogram = ee.Dictionary(
                img.reduceRegion(
                    reducer=ee.Reducer.frequencyHistogram(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=max_pixels,
                    bestEffort=True,
                ).get(year_str)
            )

            codes = histogram.keys()
            counts = ee.Array(histogram.values())
            total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
            modal_index = counts.argmax().get(0)

            modal_code = ee.Algorithms.If(codes.size().gt(0), ee.Number.parse(codes.get(modal_index)), 0)
            modal_count = ee.Algorithms.If(codes.size().gt(0), counts.get([modal_index]), 0)
            percent = ee.Algorithms.If(codes.size().gt(0), ee.Number(modal_count).divide(total), 0)

            return ee.Feature(
                None,
                {
                    "00_indexnum": indexnum,
                    "01_timestamp": ee.String(year_str).cat("-01-01"),
                    "10_class": modal_code,
                    "11_percent": percent,
                    "99_validPixelsCount": ee.Algorithms.If(codes.size().gt(0), total, 0),
                },
            )

        features = ee.FeatureCollection(years.map(_feat_for_year))
        return features.filter(ee.Filter.gt("99_validPixelsCount", 0))

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class MapBiomasC11(_AnnualModalClass):
    """MapBiomas Brazil Collection 11 LULC — annual land cover, 1985 to 2025, 30 m.

    The Collection 11 counterpart of :class:`~agrigee_lite.sat.mapbiomas.MapBiomas`,
    which stays on Collection 10. For each year in the requested range,
    ``compute()`` returns:

    - ``class`` — the modal (most frequent) land-use/cover class code (int).
    - ``percent`` — fraction of pixels agreeing with the modal class (0-1).

    ``validPixelsCount`` carries the classified pixel count the fraction was
    computed over.

    The ``classes`` attribute maps integer codes to ``{"label", "color"}``
    dicts (e.g. ``39 -> {"label": "Soybean", ...}``).

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction. Helps remove
        classification noise near geometry edges.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    Coverage is Brazil only. Collection 11 extends the series to 2025 and
    revises earlier years, so its values are not identical to Collection 10's
    for the same year — that is why both are available rather than one being
    an in-place upgrade.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__(border_pixels_to_erode, min_area_to_keep_border)
        self.imageAsset = f"{_COLLECTION11}/mapbiomas_brazil_collection11_coverage_v3"
        self.pixelSize = 30
        self.firstYear = 1985
        self.lastYear = 2025
        self.startDate = "1985-01-01"
        self.endDate = "2025-12-31"
        self.shortName = "mapbiomasc11majclass"

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
            13: {"label": "Other non Forest Formations", "color": "#d89f5c"},
            14: {"label": "Farming", "color": "#ffefc3"},
            15: {"label": "Pasture", "color": "#edde8e"},
            18: {"label": "Agriculture", "color": "#e974ed"},
            19: {"label": "Temporary Crop", "color": "#c27ba0"},
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
            62: {"label": "Cotton", "color": "#ff69b4"},
        }


class MapBiomasSecondCrop(_AnnualClassFractions):
    """MapBiomas Collection 11 second crop (safrinha) — annual, 2000 to 2025, 30 m.

    The MapBiomas Agriculture module maps what is grown in the *second* cycle
    of a crop year, after the main harvest. For each year in the requested
    range, ``compute()`` returns the fraction of mapped pixels in each outcome:

    - ``noSecondCrop`` — mapped, but nothing grown in the second cycle.
    - ``cornSecondCrop`` — corn, by far the dominant safrinha crop.
    - ``cottonSecondCrop`` — cotton.
    - ``otherTemporarySecondCrop`` — other temporary crops.

    The four fractions sum to 1, so the dominant outcome is their ``argmax``.
    ``validPixelsCount`` carries the mapped pixel count they were computed over.

    ``noSecondCrop`` is a published value rather than an absence, which is what
    makes this product usable: a year where the field was mapped and left fallow
    is distinguishable from a field the product does not cover. Geometries with
    no mapped pixels at all yield no row rather than a row of zeros.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    This product is **not national**. Only ten states are mapped: BA, GO, MA,
    MG, MS, MT, PI, PR, SP and TO. The cotton subclass is narrower still —
    mapped only in MT, MS, GO, MA, TO, PI and BA, while in SP, PR and MG cotton
    is folded into other temporary crops. A cotton-versus-other comparison is
    therefore not valid across state lines.

    Pixels are restricted to those whose first cycle was soybean, cotton or
    another temporary crop in the Collection 11 land cover map, so a pasture or
    forest pixel is never mapped here at all.

    The underlying crop year runs from 1 September of the previous year to 31
    August. Timestamps stay on ``<year>-01-01`` to line up with the other annual
    sources in this library, so the label is the harvest year, not the window.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__(border_pixels_to_erode, min_area_to_keep_border)
        self.imageAsset = f"{_COLLECTION11}/mapbiomas_brazil_collection11_agriculture_second_crop_v1"
        self.pixelSize = 30
        self.firstYear = 2000
        self.lastYear = 2025
        self.startDate = "2000-01-01"
        self.endDate = "2025-12-31"
        self.shortName = "mapbiomassecondcrop"

        self.classes = {
            0: {"label": "No Second Crop", "color": "#e5e5e5"},
            1: {"label": "Corn", "color": "#ffd400"},
            41: {"label": "Other Temporary Crops", "color": "#f54ca9"},
            62: {"label": "Cotton", "color": "#ff69b4"},
        }
        self.classColumns = {
            0: "10_noSecondCrop",
            1: "11_cornSecondCrop",
            62: "12_cottonSecondCrop",
            41: "13_otherTemporarySecondCrop",
        }
        self.selectedBands = [("", column) for column in self.classColumns.values()]
        self.toDownloadSelectors = list(self.classColumns.values())


class MapBiomasCropCycles(_AnnualClassFractions):
    """MapBiomas Collection 11 crop cycles — cycles per year, 2017 to 2025, 10 m.

    How many crop cycles a pixel carried in a year: one (a single harvest), two
    (safra plus safrinha), or three or more. For each year, ``compute()``
    returns the fraction of cropped pixels in each:

    - ``oneCycle`` — a single cycle.
    - ``twoCycles`` — two cycles.
    - ``threeOrMoreCycles`` — three or more; the source caps its legend here.

    The three fractions sum to 1 and ``validPixelsCount`` carries the cropped
    pixel count behind them.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction. One pixel is 10 m
        here, not 30 m, so this erodes far less ground than the same value on
        the other MapBiomas sources.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    Unlike the rest of the MapBiomas sources in this library, this product is
    Sentinel-derived: **10 m** pixels, and only **2017 onwards**. It does not
    extend back to 1985, so pairing it with a longer series leaves the early
    years empty.

    The crop year runs 1 September to 31 August; timestamps stay on
    ``<year>-01-01`` for consistency with the other annual sources.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__(border_pixels_to_erode, min_area_to_keep_border)
        self.imageAsset = f"{_COLLECTION11}/mapbiomas_brazil_collection11_agriculture_number_cycles_v1"
        self.pixelSize = 10
        self.firstYear = 2017
        self.lastYear = 2025
        self.startDate = "2017-01-01"
        self.endDate = "2025-12-31"
        self.shortName = "mapbiomascropcycles"

        self.classes = {
            1: {"label": "One Cycle", "color": "#fee08b"},
            2: {"label": "Two Cycles", "color": "#66bd63"},
            3: {"label": "Three or More Cycles", "color": "#1a9850"},
        }
        self.classColumns = {
            1: "10_oneCycle",
            2: "11_twoCycles",
            3: "12_threeOrMoreCycles",
        }
        self.selectedBands = [("", column) for column in self.classColumns.values()]
        self.toDownloadSelectors = list(self.classColumns.values())


class MapBiomasPastureVigorC11(_AnnualClassFractions):
    """MapBiomas Collection 11 pasture vigor — annual, 2000 to 2025, 30 m.

    The Collection 11 counterpart of
    :class:`~agrigee_lite.sat.mapbiomas.MapBiomasPastureVigor`, which stays on
    Collection 10 and ends in 2024. For each year, ``compute()`` returns the
    fraction of pasture pixels in each vigour class:

    - ``lowVigor`` — reduced forage production, evidence of severe degradation.
    - ``mediumVigor``
    - ``highVigor``

    The three fractions sum to 1 over that year's pasture pixels, so the
    dominant condition is their ``argmax``, and ``validPixelsCount`` carries the
    pasture pixel count. Years where the geometry holds no pasture yield no row
    rather than zeros.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    Coverage is Brazil only, and only where MapBiomas maps pasture. The
    underlying signal is a MODIS EVI trend, seasonally decomposed and normalised
    per biome, published at 30 m — so the vigour classes carry MODIS-scale
    detail regardless of the pixel size.

    The colours in ``classes`` are an AgriGEE.lite red-yellow-green ramp, not an
    official MapBiomas palette.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__(border_pixels_to_erode, min_area_to_keep_border)
        self.imageAsset = f"{_COLLECTION11}/mapbiomas_brazil_collection11_pasture_vigor_v1"
        self.pixelSize = 30
        self.firstYear = 2000
        self.lastYear = 2025
        self.startDate = "2000-01-01"
        self.endDate = "2025-12-31"
        self.shortName = "mapbiomaspasturevigorc11"

        self.classes = {
            1: {"label": "Low Vigor", "color": "#d73027"},
            2: {"label": "Medium Vigor", "color": "#fee08b"},
            3: {"label": "High Vigor", "color": "#1a9850"},
        }
        self.classColumns = {
            1: "10_lowVigor",
            2: "11_mediumVigor",
            3: "12_highVigor",
        }
        self.selectedBands = [("", column) for column in self.classColumns.values()]
        self.toDownloadSelectors = list(self.classColumns.values())


class MapBiomasIrrigation(_AnnualClassFractions):
    """MapBiomas Collection 11 irrigation systems — annual, 1985 to 2025, 30 m.

    Irrigated area by system type. For each year, ``compute()`` returns the
    fraction of *irrigated* pixels in each system:

    - ``centralPivot`` — centre-pivot systems.
    - ``otherIrrigation`` — other irrigation systems.
    - ``irrigatedRice`` — flooded rice.

    The three fractions sum to 1 and ``validPixelsCount`` carries the irrigated
    pixel count.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    The denominator is irrigated pixels, not the geometry: this answers "of the
    irrigated area, what kind" and not "how much of this field is irrigated".
    Use ``validPixelsCount`` against the geometry's own pixel count for the
    latter. A geometry with no irrigation yields no row at all.

    The mapping of codes 1, 2 and 3 to pivot, other and rice is inferred from
    the module's structure and from where each code appears — MapBiomas has not
    published this legend. Treat the labels, not the fractions, as the
    uncertain part.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__(border_pixels_to_erode, min_area_to_keep_border)
        self.imageAsset = f"{_COLLECTION11}/mapbiomas_brazil_collection11_agriculture_irrigation_systems_v1"
        self.pixelSize = 30
        self.firstYear = 1985
        self.lastYear = 2025
        self.startDate = "1985-01-01"
        self.endDate = "2025-12-31"
        self.shortName = "mapbiomasirrigation"

        self.classes = {
            1: {"label": "Central Pivot", "color": "#1a9850"},
            2: {"label": "Other Irrigation", "color": "#66bd63"},
            3: {"label": "Irrigated Rice", "color": "#2532e4"},
        }
        self.classColumns = {
            1: "10_centralPivot",
            2: "11_otherIrrigation",
            3: "12_irrigatedRice",
        }
        self.selectedBands = [("", column) for column in self.classColumns.values()]
        self.toDownloadSelectors = list(self.classColumns.values())


class MapBiomasPastureAge(DataSourceSatellite):
    """MapBiomas Collection 11 pasture age — years since establishment, 1985 to 2025, 30 m.

    How long a pasture pixel has been pasture. For each year in the requested
    range, ``compute()`` returns:

    - ``meanPastureAge`` — mean age in years over pixels with a known age.
    - ``agedPixelsFraction`` — fraction of pasture pixels that have a known age.
    - ``undatedPixelsFraction`` — fraction that do not.

    ``validPixelsCount`` carries the total pasture pixel count.

    The source encodes age as ``200 + years``, so 2025's oldest possible value
    is ``240`` — forty years after the 1985 start of the series. Pixels that
    were already pasture when the series begins carry the separate code ``100``
    and have no measurable age: their true age is censored, not zero. They are
    excluded from ``meanPastureAge`` and reported through
    ``undatedPixelsFraction`` instead, because averaging them in as zeros would
    make the oldest pastures look like the newest.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    Coverage is Brazil only, and only where MapBiomas maps pasture. A high
    ``undatedPixelsFraction`` means long-established pasture rather than missing
    data.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__()
        self.imageAsset = f"{_COLLECTION11}/mapbiomas_brazil_collection11_pasture_age_v1"
        self.pixelSize = 30
        self.firstYear = 1985
        self.lastYear = 2025
        self.startDate = "1985-01-01"
        self.endDate = "2025-12-31"
        self.shortName = "mapbiomaspastureage"
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        # The source offsets age by 200 so that it can carry the undated code in
        # the same band; 100 marks pasture already present when the series began.
        self.ageOffset = 200
        self.undatedCode = 100

        self.selectedBands = [
            ("", "10_meanPastureAge"),
            ("", "11_agedPixelsFraction"),
            ("", "12_undatedPixelsFraction"),
        ]
        self.toDownloadSelectors = [
            "10_meanPastureAge",
            "11_agedPixelsFraction",
            "12_undatedPixelsFraction",
        ]

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

        source_image = ee.Image(self.imageAsset)

        subsampling_max_pixels_: float = subsampling_max_pixels if subsampling_max_pixels is not None else 1e8
        max_pixels = ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels_, self.pixelSize)

        indexnum = ee.Feature(ee_feature).get("0")
        years = ee_years_in_coverage(ee.Feature(ee_feature), self.firstYear, self.lastYear)

        def _feat_for_year(year: ee.Number) -> ee.Feature:
            year_str = ee.Number(year).toInt().format()
            band_in = ee.String("classification_").cat(year_str)
            img = source_image.select([band_in], [year_str])

            aged = img.gte(self.ageOffset)
            ages = img.updateMask(aged).subtract(self.ageOffset)

            aged_stats = ee.Dictionary(
                ages.reduceRegion(
                    reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=max_pixels,
                    bestEffort=True,
                )
            )
            aged_count = ee.Number(aged_stats.get(ee.String(year_str).cat("_count"), 0))
            mean_age = aged_stats.get(ee.String(year_str).cat("_mean"), 0)

            undated_count = ee.Number(
                img.eq(self.undatedCode)
                .selfMask()
                .reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=max_pixels,
                    bestEffort=True,
                )
                .get(year_str)
            )

            pasture_pixels = aged_count.add(undated_count)

            return ee.Feature(
                None,
                {
                    "00_indexnum": indexnum,
                    "01_timestamp": ee.String(year_str).cat("-01-01"),
                    "10_meanPastureAge": ee.Algorithms.If(aged_count.gt(0), mean_age, 0),
                    "11_agedPixelsFraction": ee.Algorithms.If(
                        pasture_pixels.gt(0), aged_count.divide(pasture_pixels), 0
                    ),
                    "12_undatedPixelsFraction": ee.Algorithms.If(
                        pasture_pixels.gt(0), undated_count.divide(pasture_pixels), 0
                    ),
                    "99_validPixelsCount": pasture_pixels,
                },
            )

        features = ee.FeatureCollection(years.map(_feat_for_year))
        return features.filter(ee.Filter.gt("99_validPixelsCount", 0))

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class MapBiomasMining(_AnnualModalClass):
    """MapBiomas Collection 11 mined substance — annual, 1985 to 2025, 30 m.

    Mining areas classified by activity scale and by the substance extracted.
    For each year, ``compute()`` returns:

    - ``class`` — the modal mined-substance code (int).
    - ``percent`` — fraction of mining pixels agreeing with it (0-1).

    ``validPixelsCount`` carries the mining pixel count. A geometry with no
    mining yields no row.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    ``classes`` is deliberately empty. The codes are four-digit and
    hierarchical — activity scale, substance category and specific mineral,
    following the Brazilian Geological Service (CPRM/SGB) classification — but
    MapBiomas has not yet published the code table, stating only that it will be
    released in due course. The codes returned here are the source's own, and
    can be joined against that table once it exists. Labelling them from
    inference would be guesswork.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__(border_pixels_to_erode, min_area_to_keep_border)
        self.imageAsset = f"{_COLLECTION11}/mapbiomas_brazil_collection11_mining_substances_v1"
        self.pixelSize = 30
        self.firstYear = 1985
        self.lastYear = 2025
        self.startDate = "1985-01-01"
        self.endDate = "2025-12-31"
        self.shortName = "mapbiomasmining"
        self.classes: dict[int, dict[str, str]] = {}


class MapBiomasDeforestationSecondaryVegetation(_AnnualModalClass):
    """MapBiomas Collection 11 deforestation and secondary vegetation — annual, 1987 to 2025, 30 m.

    Tracks primary vegetation loss and secondary vegetation dynamics. For each
    year, ``compute()`` returns:

    - ``class`` — the modal code (int).
    - ``percent`` — fraction of pixels agreeing with it (0-1).

    ``validPixelsCount`` carries the classified pixel count.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    The series starts in 1987 rather than 1985: the product needs preceding
    years to establish what changed.

    ``classes`` is deliberately empty. The product publishes seven codes, and
    MapBiomas has not released a table saying which is which — not in the
    Collection 10 legend document, which covers only the land cover legend, nor
    in their public repositories. The codes are returned as the source gives
    them.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__(border_pixels_to_erode, min_area_to_keep_border)
        self.imageAsset = f"{_COLLECTION11}/mapbiomas_brazil_collection11_deforestation_secondary_vegetation_v5"
        self.pixelSize = 30
        self.firstYear = 1987
        self.lastYear = 2025
        self.startDate = "1987-01-01"
        self.endDate = "2025-12-31"
        self.shortName = "mapbiomasdeforestationsecveg"
        self.classes: dict[int, dict[str, str]] = {}


class MapBiomasCropCyclesMean(SingleImageSatellite):
    """MapBiomas Collection 11 mean crop cycles — one static image, 10 m.

    The mean number of crop cycles per pixel across 2017-2025, published as a
    single band rather than a yearly series. ``compute()`` returns one row with:

    - ``meanCropCycles`` — mean cycles per year over cropped pixels.

    ``validPixelsCount`` carries the cropped pixel count behind it.

    Values are continuous, not the 1/2/3 of the annual product: a pixel that
    carried two cycles in most years and one in the rest lands between them.
    That is the point of the layer — it separates reliably double-cropped ground
    from ground that only occasionally manages a safrinha, which no single year
    can show.

    Parameters
    ----------
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction. A pixel is 10 m here.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m2).

    Notes
    -----
    Sentinel-derived at 10 m, summarising 2017-2025. Being a single image it
    carries no date, so it is exposed the same way the DEM and soil sources are.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__()
        self.imageName = f"{_COLLECTION11}/mapbiomas_brazil_collection11_agriculture_number_cycles_mean_v1"
        self.pixelSize = 10
        self.startDate = "1900-01-01"
        self.endDate = "2050-01-01"
        self.shortName = "mapbiomascropcyclesmean"
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.selectedBands = [("", "10_meanCropCycles")]
        self.toDownloadSelectors = ["10_meanCropCycles"]

    def image(self, ee_feature: ee.Feature) -> ee.Image:
        return ee.Image(self.imageName).select(["cycles_mean"], ["mean_crop_cycles"])

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

        subsampling_max_pixels_: float = subsampling_max_pixels if subsampling_max_pixels is not None else 1e8
        max_pixels = ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels_, self.pixelSize)

        cycles = self.image(ee_feature)

        stats = ee.Dictionary(
            cycles.reduceRegion(
                reducer=ee.Reducer.mean().combine(ee.Reducer.count(), sharedInputs=True),
                geometry=ee_geometry,
                scale=self.pixelSize,
                maxPixels=max_pixels,
                bestEffort=True,
            )
        )
        cropped_pixels = ee.Number(stats.get("mean_crop_cycles_count", 0))

        return ee.FeatureCollection([
            ee.Feature(
                None,
                {
                    "00_indexnum": ee.Feature(ee_feature).get("0"),
                    "10_meanCropCycles": ee.Algorithms.If(
                        cropped_pixels.gt(0), stats.get("mean_crop_cycles_mean", 0), 0
                    ),
                    "99_validPixelsCount": cropped_pixels,
                },
            )
        ])

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
