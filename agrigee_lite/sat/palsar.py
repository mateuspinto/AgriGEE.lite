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
from agrigee_lite.sat.abstract_satellite import RadarSatellite


class PALSAR2ScanSAR(RadarSatellite):
    """ALOS PALSAR-2 ScanSAR L2.2 — from 2014-08-04 to present, ~25 m resolution, ~14-day revisit.

    L-band SAR from JAXA's ALOS-2 satellite.  L-band (longer wavelength than
    Sentinel-1's C-band) penetrates vegetation canopy better, making it useful
    for forest structure and biomass studies.  Band values are backscatter
    in dB after DN→power→dB conversion.

    Available bands: ``hh`` (horizontal–horizontal), ``hv`` (horizontal–vertical).

    Parameters
    ----------
    bands : set of str, optional
        Subset of available bands.  Defaults to ``{"hh", "hv"}``.
    indices : set of str, optional
        Radar indices to compute (e.g. ``{"hhhv"}``).
    use_quality_mask : bool, default True
        Apply the MSK bitmask (bits 0–2) to retain only valid pixels (value 1).
    min_valid_pixel_count : int, default 20
        Images with fewer valid pixels over the ROI are discarded.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 35_000
        Skip border erosion for geometries smaller than this area (m²).

    Notes
    -----
    Coverage is primarily Japan and selected global observation areas; it is
    not fully global like Sentinel-1.
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_quality_mask: bool = True,
        min_valid_pixel_count: int = 20,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 35000,
    ):
        bands_: list[str] = sorted({"hh", "hv"}) if bands is None else sorted(bands)

        indices_: list[str] = [] if indices is None else sorted(indices)

        super().__init__()

        self.imageCollectionName: str = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
        self.pixelSize: int = 25
        self.startDate: str = "2014-08-04"
        self.endDate: str = "2050-01-01"
        self.shortName: str = "palsar2"

        self.availableBands: dict[str, str] = {"hh": "HH", "hv": "HV"}

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands_)]

        self.selectedIndices = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices_)
        ]

        self.use_quality_mask = use_quality_mask
        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.toDownloadSelectors = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

    @staticmethod
    def _mask_quality(img: ee.Image) -> ee.Image:
        """
        Apply MSK quality mask to exclude invalid data.

        MSK bits 0-2 indicate data quality:
            1 = valid data
            5 = invalid

        Parameters
        ----------
        img : ee.Image

        Returns
        -------
        ee.Image
        """
        mask = img.select("MSK")
        quality = mask.bitwiseAnd(0b111)
        valid = quality.eq(1)
        return img.updateMask(valid)

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_start = ee_feature.get("s")
        ee_end = ee_feature.get("e")

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start, ee_end))

        palsar_img = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        if self.use_quality_mask:
            palsar_img = palsar_img.map(self._mask_quality)

        palsar_img = palsar_img.select(
            [self.availableBands[b] for b, _ in self.selectedBands], [b for b, _ in self.selectedBands]
        )

        palsar_img = palsar_img.map(
            lambda img: ee.Image(img).addBands(
                ee.Image(img).pow(ee.Number(2)).log10().multiply(ee.Number(10)).subtract(ee.Number(83)),
                overwrite=True,
            )
        )

        if self.selectedIndices:
            palsar_img = palsar_img.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        palsar_img = palsar_img.select(
            [natural_band_name for natural_band_name, _ in self.selectedBands]
            + [indice_name for _, indice_name, _ in self.selectedIndices],
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

        palsar_img = ee_filter_img_collection_invalid_pixels(palsar_img, ee_geometry, self.pixelSize, 20)

        return palsar_img

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

        palsar_img = self.imageCollection(ee_feature)

        features = palsar_img.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )

        return features
