from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_add_indexes_to_image,
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import OpticalSatellite


def remove_l_toa_tough_clouds(img: ee.Image, filter_strength: int = 15) -> ee.Image:
    img = ee.Image(img)
    img = ee.Algorithms.Landsat.simpleCloudScore(img)

    mask = img.select(["cloud"]).lte(filter_strength)
    img = img.updateMask(mask)
    return img.select(img.bandNames().remove("cloud"))


def ee_l_mask(img: ee.Image) -> ee.Image:
    qa = img.select("cloudq")
    mask = (
        qa.bitwiseAnd(1 << 3)
        .And(qa.bitwiseAnd(1 << 8).Or(qa.bitwiseAnd(1 << 9)))
        .Or(qa.bitwiseAnd(1 << 1))
        .Or(qa.bitwiseAnd(1 << 4).And(qa.bitwiseAnd(1 << 10).Or(qa.bitwiseAnd(1 << 11))))
        .Or(qa.bitwiseAnd(1 << 5))
        .Or(qa.bitwiseAnd(1 << 7))
        .Or(qa.bitwiseAnd(1 << 2))
    )

    return img.updateMask(mask.Not()).select(img.bandNames().remove("cloudq"))


def ee_l_apply_sr_scale_factors(img: ee.Image) -> ee.Image:
    img = ee.Image(img)
    optical_bands = img.select("SR_B.").multiply(0.0000275).add(-0.2)
    # thermal_bands = img.select("ST_B6").multiply(0.00341802).add(149.0)
    return img.addBands(optical_bands, None, True)  # .addBands(thermal_bands, None, True)


class AbstractLandsat(OpticalSatellite):
    """Shared implementation for all Landsat missions (Collection 2).

    Handles band selection, cloud masking via the ``QA_PIXEL`` bitmask,
    optional TOA simple-cloud-score filter, SR scale-factor application, and
    optional pan-sharpening.  Concrete subclasses (``Landsat5`` through
    ``Landsat9``) set the sensor-specific band names and date range.

    Parameters
    ----------
    sensor_code : str
        USGS sensor code used to build the GEE collection path
        (e.g. ``"LC08"`` for Landsat 8).
    toa_band_map, sr_band_map : dict[str, str]
        Mappings from friendly band names (e.g. ``"nir"``) to GEE band names
        for TOA and SR products respectively.
    short_base : str
        Short identifier prefix (e.g. ``"l8"``); ``"sr"`` is appended when
        ``use_sr=True``.
    start_date, end_date : str
        ISO-8601 bounds of the sensor's valid acquisition period.
    bands : set of str or None
        Subset of bands to include.  Defaults to all 6 VNIR/SWIR bands for SR
        or all 7 (including ``"pan"``) for TOA.
    indices : set of str or None
        Spectral indices to compute on top of the raw bands.  Must be computable
        from the selected bands (see ``availableIndices``).
    use_sr : bool
        ``True`` (default) uses atmospherically-corrected Surface Reflectance
        (SR) — values are actual ground reflectance in 0–1.
        ``False`` uses Top-of-Atmosphere (TOA) reflectance — values still
        contain atmospheric effects but have wider historical coverage.
    tier : int
        Data quality tier.  Tier 1 has the best geometric accuracy and is
        suitable for time-series analysis.  Tier 2 has looser accuracy
        requirements.
    use_cloud_mask : bool
        Apply the QA_PIXEL bitmask to remove clouds, cloud shadows, cirrus,
        dilated clouds, and saturated pixels.
    min_valid_pixel_count : int
        Images with fewer valid pixels than this threshold over the ROI are
        discarded entirely.
    toa_cloud_filter_strength : int
        0–100 score threshold for the Landsat Simple Cloud Score applied to
        TOA images (lower = stricter).  Ignored when ``use_sr=True``.
    border_pixels_to_erode : float
        Inward buffer applied to the geometry before pixel extraction, in
        multiples of ``pixelSize``.  Reduces edge artefacts from mixed pixels.
    min_area_to_keep_border : int
        Minimum geometry area (m²) required to apply the border erosion.
        Smaller geometries are used as-is to avoid discarding them entirely.
    use_pan_sharpening : bool
        Merge the 15 m panchromatic band with RGB via HSV pan-sharpening.
        Only available for TOA products (``use_sr=False``); raises
        ``ValueError`` if combined with SR.
    """

    _DEFAULT_BANDS_BOA: set[str] = {  # noqa: RUF012
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
    }

    _DEFAULT_BANDS_TOA: set[str] = {  # noqa: RUF012
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
        "pan",
    }

    def __init__(
        self,
        *,
        sensor_code: str,  # e.g. "LT05"
        toa_band_map: dict[str, str],
        sr_band_map: dict[str, str],
        short_base: str,  # e.g. "l5"
        start_date: str,  # sensor-specific
        end_date: str,  # sensor-specific
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
        toa_cloud_filter_strength: int = 15,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
        use_pan_sharpening: bool = False,
    ) -> None:
        super().__init__()

        if use_sr and use_pan_sharpening:
            raise ValueError("Pan-sharpening is only available for TOA products (use_sr=False).")  # noqa: TRY003

        bands_: list[str] = (
            (sorted(self._DEFAULT_BANDS_BOA) if use_sr else sorted(self._DEFAULT_BANDS_TOA))
            if bands is None
            else sorted(bands)
        )

        if use_pan_sharpening and "pan" not in bands_:
            raise ValueError("When using pan-sharpening, the 'pan' band must be included in the selected bands.")  # noqa: TRY003

        indices_: list[str] = [] if indices is None else sorted(indices)

        self.useSr = use_sr
        self.tier = tier
        self.pixelSize: int = 15 if use_pan_sharpening else 30

        self.startDate: str = start_date
        self.endDate: str = end_date

        suffix = "L2" if use_sr else "TOA"
        self.imageCollectionName = f"LANDSAT/{sensor_code}/C02/T{tier}_{suffix}"
        self.shortName: str = f"{short_base}sr" if use_sr else short_base

        self.availableBands = sr_band_map if use_sr else toa_band_map
        self.availableBands["cloudq"] = "QA_PIXEL"

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands_)]

        self.selectedIndices = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices_)
        ]

        self.useCloudMask = use_cloud_mask
        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode
        self.usePanSharpening = use_pan_sharpening
        self.toaCloudFilterStrength = toa_cloud_filter_strength

        self.toDownloadSelectors = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

    def ee_l_pan_sharpen(self, image: ee.Image, geometry: ee.Geometry) -> ee.Image:
        pan = image.select("pan").clip(geometry)

        rgb = image.select(["red", "green", "blue"]).reproject(crs=pan.projection())
        hsv = rgb.rgbToHsv()

        sharpened = ee.Image.cat([hsv.select("hue"), hsv.select("saturation"), pan]).hsvToRgb()

        return image.addBands(sharpened, ["red", "green", "blue"], overwrite=True).mask(image.select("red").mask())

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        geom = ee_feature.geometry()
        ee_filter = ee.Filter.And(
            ee.Filter.bounds(geom),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        col = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        if self.useSr:
            col = col.map(ee_l_apply_sr_scale_factors)

        if not self.useSr and self.useCloudMask:
            col = col.map(partial(remove_l_toa_tough_clouds, filter_strength=self.toaCloudFilterStrength))

        col = col.select(list(self.availableBands.values()), list(self.availableBands.keys()))

        if self.usePanSharpening:
            col = col.map(partial(self.ee_l_pan_sharpen, geometry=geom))

        if self.useCloudMask:
            col = col.map(ee_l_mask)

        if self.selectedIndices:
            col = col.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        col = col.select(
            [natural_band_name for natural_band_name, _ in self.selectedBands]
            + [indice_name for _, indice_name, _ in self.selectedIndices],
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

        col = ee_filter_img_collection_invalid_pixels(col, geom, self.pixelSize, self.minValidPixelCount)
        return ee.ImageCollection(col)

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

        col = self.imageCollection(ee_feature)
        features = col.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )
        return features


class Landsat5(AbstractLandsat):
    """Landsat 5 TM — global coverage from 1984-03-01 to 2013-05-05, 30 m resolution, 16-day revisit.

    Available bands: ``blue``, ``green``, ``red``, ``nir``, ``swir1``, ``swir2``.

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands to download.  Defaults to all six.
    indices : set of str, optional
        Spectral indices to compute (e.g. ``{"ndvi", "evi2"}``).  Only indices
        whose required bands are all in ``bands`` can be requested.
    use_sr : bool, default True
        Use atmospherically-corrected Surface Reflectance (SR) products.
        Set to ``False`` for Top-of-Atmosphere (TOA) reflectance, which
        extends coverage to 2013-05-05 (SR ends 2012-05-05).
    tier : int, default 1
        Collection tier.  Tier 1 has the highest geometric accuracy and is
        recommended for time-series analysis.
    use_cloud_mask : bool, default True
        Apply the USGS QA_PIXEL bitmask to remove clouds, shadows, and
        cirrus.  Disabling this delivers more images but with much more noise.
    min_valid_pixel_count : int, default 12
        Discard any image whose valid-pixel count over the ROI is below this.
    toa_cloud_filter_strength : int, default 15
        Additional cloud-score threshold for TOA imagery (0–100, lower =
        stricter).  Ignored when ``use_sr=True``.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths applied before extraction, to avoid
        mixed-pixel artefacts at geometry edges.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
        toa_cloud_filter_strength: int = 15,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ):
        toa = {"blue": "B1", "green": "B2", "red": "B3", "nir": "B4", "swir1": "B5", "swir2": "B7"}
        sr = {
            "blue": "SR_B1",
            "green": "SR_B2",
            "red": "SR_B3",
            "nir": "SR_B4",
            "swir1": "SR_B5",
            "swir2": "SR_B7",
        }
        super().__init__(
            indices=indices,
            sensor_code="LT05",
            toa_band_map=toa,
            sr_band_map=sr,
            short_base="l5",
            start_date="1984-03-01",
            end_date="2013-05-05",
            bands=bands,
            use_sr=use_sr,
            tier=tier,
            use_cloud_mask=use_cloud_mask,
            min_valid_pixel_count=min_valid_pixel_count,
            toa_cloud_filter_strength=toa_cloud_filter_strength,
            border_pixels_to_erode=border_pixels_to_erode,
            min_area_to_keep_border=min_area_to_keep_border,
            use_pan_sharpening=False,
        )


class Landsat7(AbstractLandsat):
    """Landsat 7 ETM+ — global coverage from 1999-04-15 to 2022-04-06, 30 m resolution, 16-day revisit.

    Available bands: ``blue``, ``green``, ``red``, ``nir``, ``swir1``, ``swir2``, ``pan``.

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands to download.  Defaults to the six VNIR/SWIR
        bands (``pan`` excluded from the SR default).
    indices : set of str, optional
        Spectral indices to compute (e.g. ``{"ndvi"}``).
    use_sr : bool, default True
        Surface Reflectance products (``SR_B*``).  Set to ``False`` for TOA.
    tier : int, default 1
        Collection tier (1 = best geometric accuracy).
    use_cloud_mask : bool, default True
        Apply QA_PIXEL bitmask cloud removal.
    min_valid_pixel_count : int, default 12
        Images below this valid-pixel count over the ROI are discarded.
    toa_cloud_filter_strength : int, default 15
        Additional cloud-score filter for TOA imagery (lower = stricter).
    border_pixels_to_erode : float, default 1
        Inward buffer (in pixel-widths) before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).
    use_pan_sharpening : bool, default False
        HSV pan-sharpening of RGB using the 15 m panchromatic band.
        Requires ``use_sr=False`` and ``"pan"`` in ``bands``.

    Notes
    -----
    From 2003-05-31 onwards, the ETM+ scan-line corrector (SLC) failed,
    introducing data gaps in a striped pattern over ~22 % of each scene.
    Time-series analyses requiring gapless images should prefer Landsat 8/9
    for post-2013 data.
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
        toa_cloud_filter_strength: int = 15,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
        use_pan_sharpening: bool = False,
    ):
        toa = {"blue": "B1", "green": "B2", "red": "B3", "nir": "B4", "swir1": "B5", "swir2": "B7", "pan": "B8"}
        sr = {
            "blue": "SR_B1",
            "green": "SR_B2",
            "red": "SR_B3",
            "nir": "SR_B4",
            "swir1": "SR_B5",
            "swir2": "SR_B7",
        }
        super().__init__(
            indices=indices,
            sensor_code="LE07",
            toa_band_map=toa,
            sr_band_map=sr,
            short_base="l7",
            start_date="1999-04-15",
            end_date="2022-04-06",
            bands=bands,
            use_sr=use_sr,
            tier=tier,
            use_cloud_mask=use_cloud_mask,
            min_valid_pixel_count=min_valid_pixel_count,
            toa_cloud_filter_strength=toa_cloud_filter_strength,
            border_pixels_to_erode=border_pixels_to_erode,
            min_area_to_keep_border=min_area_to_keep_border,
            use_pan_sharpening=use_pan_sharpening,
        )


class Landsat8(AbstractLandsat):
    """Landsat 8 OLI/TIRS — global coverage from 2013-04-11 to present, 30 m resolution, 16-day revisit.

    Available bands: ``blue``, ``green``, ``red``, ``nir``, ``swir1``, ``swir2``, ``pan``.

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands.  Defaults to the six VNIR/SWIR bands.
    indices : set of str, optional
        Spectral indices to compute (e.g. ``{"ndvi", "evi2", "ndwi"}``).
    use_sr : bool, default True
        Surface Reflectance products.  Set to ``False`` for TOA.
    tier : int, default 1
        Collection tier (1 = best geometric accuracy).
    use_cloud_mask : bool, default True
        Apply QA_PIXEL bitmask cloud removal.
    min_valid_pixel_count : int, default 12
        Images below this valid-pixel count over the ROI are discarded.
    toa_cloud_filter_strength : int, default 15
        Additional cloud-score filter for TOA imagery.
    border_pixels_to_erode : float, default 1
        Inward buffer (in pixel-widths) before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).
    use_pan_sharpening : bool, default False
        HSV pan-sharpening of RGB using the 15 m panchromatic band.
        Requires ``use_sr=False`` and ``"pan"`` in ``bands``.
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
        toa_cloud_filter_strength: int = 15,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
        use_pan_sharpening: bool = False,
    ):
        toa = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6", "swir2": "B7", "pan": "B8"}
        sr = {
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir1": "SR_B6",
            "swir2": "SR_B7",
        }
        super().__init__(
            indices=indices,
            sensor_code="LC08",
            toa_band_map=toa,
            sr_band_map=sr,
            short_base="l8",
            start_date="2013-04-11",
            end_date="2050-01-01",
            bands=bands,
            use_sr=use_sr,
            tier=tier,
            use_cloud_mask=use_cloud_mask,
            min_valid_pixel_count=min_valid_pixel_count,
            toa_cloud_filter_strength=toa_cloud_filter_strength,
            border_pixels_to_erode=border_pixels_to_erode,
            min_area_to_keep_border=min_area_to_keep_border,
            use_pan_sharpening=use_pan_sharpening,
        )


class Landsat9(AbstractLandsat):
    """Landsat 9 OLI-2/TIRS-2 — global coverage from 2021-11-01 to present, 30 m resolution, 16-day revisit.

    Nearly identical to Landsat 8; provides mission continuity with improved radiometric performance.
    Available bands: ``blue``, ``green``, ``red``, ``nir``, ``swir1``, ``swir2``, ``pan``.

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands.  Defaults to the six VNIR/SWIR bands.
    indices : set of str, optional
        Spectral indices to compute (e.g. ``{"ndvi", "evi2"}``).
    use_sr : bool, default True
        Surface Reflectance products.  Set to ``False`` for TOA.
    tier : int, default 1
        Collection tier (1 = best geometric accuracy).
    use_cloud_mask : bool, default True
        Apply QA_PIXEL bitmask cloud removal.
    min_valid_pixel_count : int, default 12
        Images below this valid-pixel count over the ROI are discarded.
    toa_cloud_filter_strength : int, default 15
        Additional cloud-score filter for TOA imagery.
    border_pixels_to_erode : float, default 1
        Inward buffer (in pixel-widths) before extraction.
    min_area_to_keep_border : int, default 50_000
        Skip border erosion for geometries smaller than this area (m²).
    use_pan_sharpening : bool, default False
        HSV pan-sharpening of RGB using the 15 m panchromatic band.
        Requires ``use_sr=False`` and ``"pan"`` in ``bands``.
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
        toa_cloud_filter_strength: int = 15,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
        use_pan_sharpening: bool = False,
    ):
        toa = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6", "swir2": "B7", "pan": "B8"}
        sr = {
            "blue": "SR_B2",
            "green": "SR_B3",
            "red": "SR_B4",
            "nir": "SR_B5",
            "swir1": "SR_B6",
            "swir2": "SR_B7",
        }
        super().__init__(
            indices=indices,
            sensor_code="LC09",
            toa_band_map=toa,
            sr_band_map=sr,
            short_base="l9",
            start_date="2021-11-01",
            end_date="2050-01-01",
            bands=bands,
            use_sr=use_sr,
            tier=tier,
            use_cloud_mask=use_cloud_mask,
            min_valid_pixel_count=min_valid_pixel_count,
            toa_cloud_filter_strength=toa_cloud_filter_strength,
            border_pixels_to_erode=border_pixels_to_erode,
            min_area_to_keep_border=min_area_to_keep_border,
            use_pan_sharpening=use_pan_sharpening,
        )


class Landsat10(AbstractLandsat):
    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
        toa_cloud_filter_strength: int = 15,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
        use_pan_sharpening: bool = False,
    ):
        raise NotImplementedError("Landsat 10 is not yet available.")
