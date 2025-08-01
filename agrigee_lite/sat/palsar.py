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
    Satellite abstraction for ALOS PALSAR-2 ScanSAR (Level2.2).

    The PALSAR-2 onboard ALOS-2 provides L-band SAR observations.
    This class filters and preprocesses the ScanSAR (25m resolution) dataset.

    Parameters
    ----------
    bands : list of str, optional
        List of bands to select. Default is ['HH'].
    indices : list of str, optional
        List of custom radar indices (e.g. ratios). Default is empty.

    Satellite Information
    ---------------------
    +----------------------------+-----------------------------+
    | Name                       | ALOS PALSAR-2 ScanSAR       |
    | Revisit Time              | ~14 days                    |
    | Pixel Size                | ~25 meters                  |
    | Coverage                  | Japan + some global areas   |
    +----------------------------+-----------------------------+

    Collection Dates
    ----------------
    +------------------+------------+--------+
    | Collection Type  | Start Date | End    |
    +------------------+------------+--------+
    | Level 2.2 ScanSAR| 2014-08-04 | present|
    +------------------+------------+--------+

    Band Information
    ------------
    +--------+--------+--------------+----------------------+
    | Band   | Type   | Resolution   | Notes                |
    +--------+--------+--------------+----------------------+
    | HH     | L-band | ~25 m        | Horizontal transmit/receive |
    | HV     | L-band | ~25 m        | Horizontal transmit, vertical receive |
    | MSK    | Mask   | ~25 m        | Quality bitmask      |
    +--------+--------+--------------+----------------------+
    """

    def __init__(
        self,
        bands: list[str] | None = None,
        indices: list[str] | None = None,
    ):
        if bands is None:
            bands = ["hh", "hv"]

        if indices is None:
            indices = []

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

        palsar_img = (
            ee.ImageCollection(self.imageCollectionName)
            .filter(ee_filter)
            .map(self._mask_quality)
            .select([self.availableBands[b] for b, _ in self.selectedBands], [b for b, _ in self.selectedBands])
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
        reducers: list[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee_safe_remove_borders(ee_geometry, self.pixelSize, 35000)
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
