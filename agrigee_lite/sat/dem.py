import ee

from agrigee_lite.ee_utils import (
    ee_map_valid_pixels,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import SingleImageSatellite


class _BaseDEM(SingleImageSatellite):
    """Shared implementation for Digital Elevation Model (DEM) sources.

    Every concrete DEM in this module is a static, time-independent image
    that yields one row per geometry with terrain statistics.  Subclasses
    only need to declare the GEE asset and its native elevation band via the
    class attributes below — all extraction logic lives here.

    ``compute()`` produces:
    - ``elevation_mean`` — mean elevation (m above sea level).
    - ``slope_*`` — fraction of pixels in six slope classes (flat ≤3°,
      gentle 3–8°, undulating 8–20°, strong 20–45°, mountainous 45–75°,
      steep >75°).
    - ``cardinal_*`` — fraction of pixels in each of the eight compass
      aspect directions.

    Class attributes (set by subclasses)
    -------------------------------------
    imageName : str
        GEE asset id (an ``Image`` or, when ``isCollection`` is True, an
        ``ImageCollection`` that is mosaicked into a single image).
    sourceBand : str
        Name of the elevation band in the source asset.  Renamed internally
        to ``"elevation"`` so slope/aspect derivation is source-agnostic.
    isCollection : bool
        Whether ``imageName`` refers to an ``ImageCollection`` that must be
        mosaicked before use.
    noData : float or None
        Sentinel value to mask out, when the source encodes voids with a
        fixed number (e.g. ANADEM uses ``-9999``).  ``None`` skips masking.
    pixelSize : float
        Native spatial resolution in metres.
    shortName : str
        Identifier used in cache keys and output column prefixes.

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

    imageName: str = ""
    sourceBand: str = "elevation"
    isCollection: bool = False
    noData: float | None = None
    pixelSize: float = 30
    shortName: str = ""

    def __init__(
        self,
        bands: list[str] | None = None,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ):
        if bands is None:
            bands = ["elevation", "slope", "aspect"]

        super().__init__()

        # ``AbstractSatellite.__init__`` resets shortName/pixelSize as instance
        # attributes, shadowing the class attributes — restore them here.
        cls = type(self)
        self.imageName = cls.imageName
        self.sourceBand = cls.sourceBand
        self.isCollection = cls.isCollection
        self.noData = cls.noData
        self.pixelSize = cls.pixelSize
        self.shortName = cls.shortName

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
        if self.isCollection:
            raw = ee.ImageCollection(self.imageName).mosaic()
        else:
            raw = ee.Image(self.imageName)

        # Normalise the source elevation band name so terrain derivation and
        # ``compute()`` are agnostic to how each source names its band.
        elevation = raw.select([self.sourceBand], ["elevation"])

        if self.noData is not None:
            elevation = elevation.updateMask(elevation.neq(ee.Number(self.noData)))

        image = elevation

        requested_bands = [b for b, _ in self.selectedBands]

        if any(b in requested_bands for b in ["slope", "aspect"]):
            terrain = ee.Terrain.products(elevation)
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


# --------------------------------------------------------------------------- #
# Global DEMs
# --------------------------------------------------------------------------- #


class ANADEM(_BaseDEM):
    """ANADEM — Brazilian territory DEM, 30 m (FURGS / ANA, SRTM-derived)."""

    imageName = "projects/et-brasil/assets/anadem/v1"
    sourceBand = "elevation"
    isCollection = False
    noData = -9999
    pixelSize = 30
    shortName = "anadem"


class CopernicusDEM(_BaseDEM):
    """Copernicus DEM GLO-30 — global DSM, 30 m (ESA / TanDEM-X, 2011-2015)."""

    imageName = "COPERNICUS/DEM/GLO30"
    sourceBand = "DEM"
    isCollection = True
    noData = None
    pixelSize = 30
    shortName = "copdem"


class NASADEM(_BaseDEM):
    """NASADEM — reprocessed SRTM, 30 m, global 60°N–56°S (NASA / USGS, 2000)."""

    imageName = "NASA/NASADEM_HGT/001"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 30
    shortName = "nasadem"


class SRTM(_BaseDEM):
    """SRTM GL1 v003 — 1 arc-second (~30 m) global DEM, 60°N–56°S (NASA / USGS, 2000)."""

    imageName = "USGS/SRTMGL1_003"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 30
    shortName = "srtm"


class CGIARSRTM(_BaseDEM):
    """CGIAR SRTM v4 — void-filled SRTM, 90 m, global 60°N–56°S (CGIAR-CSI, 2000)."""

    imageName = "CGIAR/SRTM90_V4"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 90
    shortName = "cgiar_srtm"


class ASTERDEM(_BaseDEM):
    """ASTER GDEM v3 — stereo-optical DEM, 30 m, global 83°N–83°S (NASA / METI)."""

    imageName = "projects/sat-io/open-datasets/ASTER/GDEM"
    sourceBand = "b1"
    isCollection = False
    pixelSize = 30
    shortName = "aster"


class ALOSWorld3D(_BaseDEM):
    """ALOS AW3D30 v4.1 — PRISM-derived DSM, 30 m, global 82°N–82°S (JAXA, 2006-2011)."""

    imageName = "JAXA/ALOS/AW3D30/V4_1"
    sourceBand = "DSM"
    isCollection = True
    pixelSize = 30
    shortName = "alos_aw3d30"


class MERITDEM(_BaseDEM):
    """MERIT DEM v1.0.3 — error-removed SRTM/AW3D30, 90 m, 90°N–60°S (U. Tokyo)."""

    imageName = "MERIT/DEM/v1_0_3"
    sourceBand = "dem"
    isCollection = False
    pixelSize = 90
    shortName = "merit"


class GMTED2010(_BaseDEM):
    """GMTED2010 — multi-source global terrain, ~250 m, 84°N–56°S (USGS / NGA)."""

    imageName = "USGS/GMTED2010_FULL"
    sourceBand = "be75"
    isCollection = False
    pixelSize = 250
    shortName = "gmted2010"


class GTOPO30(_BaseDEM):
    """GTOPO30 — 30 arc-second (~1 km) global DEM (USGS EROS, 1996)."""

    imageName = "USGS/GTOPO30"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 1000
    shortName = "gtopo30"


class ETOPO1(_BaseDEM):
    """ETOPO1 — 1 arc-minute (~1.8 km) global relief incl. bathymetry (NOAA NGDC, 2008)."""

    imageName = "NOAA/NGDC/ETOPO1"
    sourceBand = "bedrock"
    isCollection = False
    pixelSize = 1800
    shortName = "etopo1"


class GLOBathy(_BaseDEM):
    """GLOBathy — max-depth bathymetry for 1.4M global lakes/reservoirs, 30 m (sat-io, 2022)."""

    imageName = "projects/sat-io/open-datasets/GLOBathy/GLOBathy_bathymetry"
    sourceBand = "b1"
    isCollection = False
    pixelSize = 30
    shortName = "globathy"


# --------------------------------------------------------------------------- #
# Regional DEMs
# --------------------------------------------------------------------------- #


class AHN2Interpolated(_BaseDEM):
    """Netherlands AHN2 — 0.5 m LiDAR DTM, void-interpolated (Rijkswaterstaat, 2007-2012)."""

    imageName = "AHN/AHN2_05M_INT"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 0.5
    shortName = "ahn2_int"


class AHN2NonInterpolated(_BaseDEM):
    """Netherlands AHN2 — 0.5 m LiDAR DTM, voids retained (Rijkswaterstaat, 2007-2012)."""

    imageName = "AHN/AHN2_05M_NON"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 0.5
    shortName = "ahn2_non"


class AHN2Raw(_BaseDEM):
    """Netherlands AHN2 — 0.5 m raw LiDAR samples (Rijkswaterstaat, 2007-2012)."""

    imageName = "AHN/AHN2_05M_RUW"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 0.5
    shortName = "ahn2_ruw"


class AHN3(_BaseDEM):
    """Netherlands AHN3 — 0.5 m LiDAR DTM (Rijkswaterstaat, 2014-2019)."""

    imageName = "AHN/AHN3"
    sourceBand = "dtm"
    isCollection = True
    pixelSize = 0.5
    shortName = "ahn3"


class AHN4(_BaseDEM):
    """Netherlands AHN4 — 0.5 m LiDAR DTM (Rijkswaterstaat, 2020-2022)."""

    imageName = "AHN/AHN4"
    sourceBand = "dtm"
    isCollection = True
    pixelSize = 0.5
    shortName = "ahn4"


class USGS3DEP1m(_BaseDEM):
    """USGS 3DEP — 1 m LiDAR DTM, partial USA coverage (USGS)."""

    imageName = "USGS/3DEP/1m"
    sourceBand = "elevation"
    isCollection = True
    pixelSize = 1
    shortName = "usgs_3dep_1m"


class NEONDEM(_BaseDEM):
    """NEON DEM — 1 m LiDAR DTM at NEON field sites, USA (NSF NEON)."""

    imageName = "projects/neon-prod-earthengine/assets/DEM/001"
    sourceBand = "DTM"
    isCollection = True
    pixelSize = 1
    shortName = "neon"


class England1mTerrain(_BaseDEM):
    """England 1 m composite DTM, 99% coverage (Environment Agency, 2000-2022)."""

    imageName = "UK/EA/ENGLAND_1M_TERRAIN/2022"
    sourceBand = "dtm"
    isCollection = False
    pixelSize = 1
    shortName = "england_1m"


class FranceRGEAlti(_BaseDEM):
    """France RGE ALTI 1 m national DTM, metropolitan France (IGN, 2009-2021)."""

    imageName = "IGN/RGE_ALTI/1M/2_0"
    sourceBand = "MNT"
    isCollection = True
    pixelSize = 1
    shortName = "france_rge_alti"


class ArcticDEMMosaic(_BaseDEM):
    """ArcticDEM Mosaic v4.1 — 2 m stereo DEM, Arctic 50–90°N (PGC / NSF, 2012-2020)."""

    imageName = "UMN/PGC/ArcticDEM/V4/2m_mosaic"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 2
    shortName = "arcticdem_mosaic"


class ArcticDEMStrips(_BaseDEM):
    """ArcticDEM Strips v3 — 2 m time-stamped stereo DEMs, Arctic (PGC / NSF, 2009-2017)."""

    imageName = "UMN/PGC/ArcticDEM/V3/2m"
    sourceBand = "elevation"
    isCollection = True
    pixelSize = 2
    shortName = "arcticdem_strips"


class REMAStrips2m(_BaseDEM):
    """REMA Strips v1 — 2 m time-stamped stereo DEMs, Antarctica (PGC / NSF, 2009-2018)."""

    imageName = "UMN/PGC/REMA/V1/2m"
    sourceBand = "elevation"
    isCollection = True
    pixelSize = 2
    shortName = "rema_strips_2m"


class REMAMosaic(_BaseDEM):
    """REMA Mosaic v1.1 — 8 m stereo DEM, Antarctica (PGC / NSF, 2009-2018)."""

    imageName = "UMN/PGC/REMA/V1_1/8m"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 8
    shortName = "rema_mosaic"


class REMAStrips8m(_BaseDEM):
    """REMA Strips v1 — 8 m time-stamped stereo DEMs, Antarctica (PGC / NSF, 2009-2018)."""

    imageName = "UMN/PGC/REMA/V1/8m"
    sourceBand = "elevation"
    isCollection = True
    pixelSize = 8
    shortName = "rema_strips_8m"


class Australia5mDEM(_BaseDEM):
    """Australia 5 m DEM — LiDAR/photogrammetry, expanding coverage (Geoscience Australia)."""

    imageName = "AU/GA/AUSTRALIA_5M_DEM"
    sourceBand = "elevation"
    isCollection = True
    pixelSize = 5
    shortName = "australia_5m"


class USGS3DEP10m(_BaseDEM):
    """USGS 3DEP — 10 m DEM, contiguous USA + Alaska + Hawaii (USGS)."""

    imageName = "USGS/3DEP/10m_collection"
    sourceBand = "elevation"
    isCollection = True
    pixelSize = 10
    shortName = "usgs_3dep_10m"


class CanadaCDEM(_BaseDEM):
    """Canadian DEM (CDEM) — ~23 m multi-source mosaic (Natural Resources Canada, 1945-2011)."""

    imageName = "NRCan/CDEM"
    sourceBand = "elevation"
    isCollection = True
    pixelSize = 23
    shortName = "cdem"


class GreenlandGIMP(_BaseDEM):
    """Greenland GIMP DEM — 30 m ASTER/SPOT-5 DEM (Ohio State / NASA, 2003-2009)."""

    imageName = "OSU/GIMP/DEM"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 30
    shortName = "gimp"


class AustraliaDEMS(_BaseDEM):
    """Australia DEM-S — ~30 m smoothed, hydrologically conditioned DEM (Geoscience Australia)."""

    imageName = "AU/GA/DEM_1SEC/v10/DEM-S"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 30
    shortName = "australia_dem_s"


class AustraliaDEMH(_BaseDEM):
    """Australia DEM-H — ~30 m hydrologically enforced DEM (Geoscience Australia)."""

    imageName = "AU/GA/DEM_1SEC/v10/DEM-H"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 30
    shortName = "australia_dem_h"


class CryoSat2Antarctica(_BaseDEM):
    """CryoSat-2 Antarctica DEM — 1 km radar-altimetry DEM (ESA / CPOM, 2010-2016)."""

    imageName = "CPOM/CryoSat2/ANTARCTICA_DEM"
    sourceBand = "elevation"
    isCollection = False
    pixelSize = 1000
    shortName = "cryosat2_antarctica"
