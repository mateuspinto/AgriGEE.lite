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
    """
    Satellite abstraction for ALOS PALSAR-2 ScanSAR (Level 2.2).

    PALSAR-2 is an L-band Synthetic Aperture Radar (SAR) sensor onboard the ALOS-2 satellite,
    operated by JAXA. This class provides preprocessing and abstraction for the Level 2.2
    ScanSAR data product with 25-meter resolution. Optionally applies the MSK quality mask.

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['hh', 'hv'].
    indices : set of str, optional
        Radar indices to compute (e.g., polarization ratios). Defaults to [].
    use_quality_mask : bool, default=True
        Whether to apply the MSK bitmask quality filter. If False, all pixels are retained,
        including those marked as low-quality or invalid.
    min_valid_pixel_count : int, default=20
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=35_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Quality Masking
    ---------------
    When `use_quality_mask=True`, the `MSK` band is used to filter out invalid pixels.
    The first 3 bits of the `MSK` band indicate data quality:
        - 1 → Valid
        - 5 → Invalid
    Only pixels with value 1 are retained.

    Satellite Information
    ---------------------
    +----------------------------+-------------------------------+
    | Field                      | Value                         |
    +----------------------------+-------------------------------+
    | Name                       | ALOS PALSAR-2 ScanSAR         |
    | Sensor                     | PALSAR-2 (L-band SAR)         |
    | Platform                   | ALOS-2                        |
    | Revisit Time               | ~14 days                      |
    | Pixel Size                 | ~25 meters                    |
    | Coverage                   | Japan + selected global areas |
    +----------------------------+-------------------------------+

    Collection Dates
    ----------------
    +----------------+-------------+------------+
    | Collection     | Start Date  | End Date   |
    +----------------+-------------+------------+
    | Level 2.2      | 2014-08-04  | present    |
    +----------------+-------------+------------+

    Band Information
    ----------------
    +-----------+---------+------------+-------------------------------------------+
    | Band Name | Type    | Resolution | Description                               |
    +-----------+---------+------------+-------------------------------------------+
    | hh        | L-band  | ~25 m      | Horizontal transmit and receive           |
    | hv        | L-band  | ~25 m      | Horizontal transmit, vertical receive     |
    | msk       | Bitmask | ~25 m      | MSK quality band (used only if enabled)   |
    +-----------+---------+------------+-------------------------------------------+

    Notes
    -----
    - Earth Engine Dataset:
        https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR-2_Level2_2_ScanSAR

    - MSK Quality Mask Details (bit pattern):
        https://www.eorc.jaxa.jp/ALOS/en/palsar_fnf/data/Format_PALSAR-2.html
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
        bands = sorted({"hh", "hv"}) if bands is None else sorted(bands)

        indices = [] if indices is None else sorted(indices)

        super().__init__()

        self.imageCollectionName: str = "JAXA/ALOS/PALSAR-2/Level2_2/ScanSAR"
        self.pixelSize: int = 25
        self.startDate: str = "2014-08-04"
        self.endDate: str = "2050-01-01"
        self.shortName: str = "palsar2"

        self.availableBands: dict[str, str] = {"hh": "HH", "hv": "HV"}

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
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
            lambda img: ee.Image(img).addBands(ee.Image(img).pow(2).log10().multiply(10).subtract(83), overwrite=True)
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
