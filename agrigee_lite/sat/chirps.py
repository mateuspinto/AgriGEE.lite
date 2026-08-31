import ee

from agrigee_lite.ee_utils import ee_get_number_of_pixels, ee_safe_remove_borders
from agrigee_lite.sat.abstract_satellite import DataSourceSatellite


class ChirpsAnnualRainfall(DataSourceSatellite):
    """CHIRPS v2.0 annual rainfall — one total per calendar year, from 1981 onwards.

    CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data) is a
    daily, quasi-global precipitation product at ~5.5 km, blending satellite
    infrared estimates with station records.  This class does not return the
    daily series: for each calendar year in the requested date range it sums the
    daily images and reduces them over the geometry, yielding one row per year.

    - ``rainfallMm`` — accumulated rainfall over the year (mm), spatially
      reduced over the geometry.

    Ten years of rainfall therefore cost ten rows rather than ~3650, which is
    what makes this usable through the SITS endpoints for trend work — the
    intended consumer is a rainfall-adjusted vegetation trend (RESTREND), where
    an annual total is the predictor and a daily series would be discarded
    immediately after being summed.

    ``validPixelsCount`` carries the pixel count the spatial reduction ran
    over, so a caller can tell a geometry smaller than one CHIRPS cell (very
    common: 5.5 km is larger than most paddocks) from a full reduction.  With
    ``bestEffort`` such a geometry still returns its containing cell's value.

    Years outside coverage are clipped away, so a wider date range simply
    yields the available years.

    Parameters
    ----------
    border_pixels_to_erode : float, default 0
        Inward buffer in pixel-widths before extraction.  Defaults to 0, unlike
        the classification products: at 5.5 km, eroding even one pixel would
        erase most agricultural geometries.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).

    Notes
    -----
    Coverage is 50°S-50°N.  Rainfall is a climate variable, not a reflectance
    band: it must never be fed to a vegetation-index computation.
    """

    def __init__(
        self,
        border_pixels_to_erode: float = 0,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__()
        self.imageCollectionAsset: str = "UCSB-CHG/CHIRPS/DAILY"
        self.pixelSize: int = 5566
        self.startDate = "1981-01-01"
        # Open-ended product, so this is the same far-future sentinel the other
        # still-operational sources use. It must not be left empty: the temporal
        # gate in download_single_sits compares date strings, and any start date
        # sorts after "", which would reject every request.
        self.endDate = "2050-01-01"
        self.firstYear = 1981
        self.lastYear = 2100  # open-ended product; the collection filter bounds it
        self.shortName = "chirpsannualrainfall"
        self.selectedBands = [
            ("precipitation", "10_rainfallMm"),
        ]
        self.selectedIndices: list = []
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode
        self.toDownloadSelectors = ["10_rainfallMm"]

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

        daily = ee.ImageCollection(self.imageCollectionAsset).select("precipitation")

        subsampling_max_pixels_: float = subsampling_max_pixels if subsampling_max_pixels is not None else 1e8
        max_pixels = ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels_, self.pixelSize)

        indexnum = ee.Feature(ee_feature).get("0")
        years = self._years_in_coverage(ee.Feature(ee_feature))

        def _feat_for_year(year: ee.Number) -> ee.Feature:
            year_number = ee.Number(year).toInt()
            year_str = year_number.format()
            start = ee.Date.fromYMD(year_number, 1, 1)
            end = start.advance(1, "year")
            window = daily.filterDate(start, end)

            # An empty year (a range reaching past the collection) sums to a
            # constant-zero image, which would read as a drought rather than as
            # missing data — filtered out below on validPixelsCount instead.
            total = window.sum().rename([year_str])

            reduced = total.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=ee_geometry,
                scale=self.pixelSize,
                maxPixels=max_pixels,
                bestEffort=True,
            )
            pixel_count = ee.Number(
                total.reduceRegion(
                    reducer=ee.Reducer.count(),
                    geometry=ee_geometry,
                    scale=self.pixelSize,
                    maxPixels=max_pixels,
                    bestEffort=True,
                ).get(year_str)
            )
            # Zero images in the window means the year is outside the archive,
            # whatever the pixel count says.
            available = window.size().gt(0)

            return ee.Feature(
                None,
                {
                    "00_indexnum": indexnum,
                    "01_timestamp": ee.String(year_str).cat("-01-01"),
                    "10_rainfallMm": reduced.get(year_str),
                    "99_validPixelsCount": ee.Algorithms.If(available, pixel_count, 0),
                },
            )

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
