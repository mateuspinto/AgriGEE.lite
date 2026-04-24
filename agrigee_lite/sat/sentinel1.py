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


class Sentinel1GRD(RadarSatellite):
    """Sentinel-1 C-band SAR GRD — global coverage from 2014-10-03 to present, ~10 m resolution.

    Unlike optical sensors, SAR works in all weather and at night.  Band values
    are backscatter intensity in decibels (dB), not reflectance.  Ascending and
    descending orbit passes image the same area from different angles, yielding
    different backscatter signatures.

    Available bands: ``vv`` (vertical–vertical), ``vh`` (vertical–horizontal).

    .. warning::
        Sentinel-1B failed in December 2021.  Since then only Sentinel-1A is
        operational, roughly halving revisit frequency (~12 days instead of ~6)
        in many regions, especially the Southern Hemisphere.

    Parameters
    ----------
    bands : set of str, optional
        Polarisation channels to include.  Defaults to ``{"vv", "vh"}``.
    indices : set of str, optional
        Radar-derived indices to compute (e.g. ``{"vhvv"}``).
    ascending : bool, default True
        Use ascending orbit passes.  Set to ``False`` for descending.
        Mixing both in the same time series introduces geometric inconsistencies.
    use_edge_mask : bool, default True
        Mask pixels below −30 dB, which typically correspond to acquisition
        edges and radar shadow/layover artefacts.
    min_valid_pixel_count : int, default 20
        Images with fewer valid pixels over the ROI are discarded.
    border_pixels_to_erode : float, default 1
        Inward buffer in pixel-widths before extraction.
    min_area_to_keep_border : int, default 35_000
        Skip border erosion for geometries smaller than this area (m²).
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        ascending: bool = True,
        use_edge_mask: bool = True,
        min_valid_pixel_count: int = 20,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 35000,
    ):
        bands_: list[str] = sorted({"vv", "vh"}) if bands is None else sorted(bands)

        indices_: list[str] = [] if indices is None else sorted(indices)

        super().__init__()

        self.ascending: bool = ascending
        self.use_edge_mask: bool = use_edge_mask
        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode
        self.imageCollectionName: str = "COPERNICUS/S1_GRD"
        self.pixelSize: int = 10

        # full mission start (S-1A launch)
        self.startDate: str = "2014-10-03"
        self.endDate: str = "2050-01-01"
        self.shortName: str = "s1a" if ascending else "s1d"

        # original → product band
        self.availableBands: dict[str, str] = {"vv": "VV", "vh": "VH"}

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands_)]

        self.selectedIndices = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices_)
        ]

        self.toDownloadSelectors = [numeral_band_name for _, numeral_band_name in self.selectedBands] + [
            numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices
        ]

    @staticmethod
    def _mask_edge(img: ee.Image) -> ee.Image:
        """
        Remove extreme low-backscatter areas (edges / layover)

        Parameters
        ----------
        img : ee.Image
            Unfiltered Sentinel-1 image

        Returns
        -------
        ee.Image
            Filtered Sentinel-1 image
        """

        edge = img.lt(ee.Number(-30.0))
        valid = img.mask().And(edge.Not())
        return img.updateMask(valid)

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_start = ee_feature.get("s")
        ee_end = ee_feature.get("e")

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start, ee_end))

        polarization_filter = ee.Filter.And(*[
            ee.Filter.listContains("transmitterReceiverPolarisation", self.availableBands[b])
            for b, _ in self.selectedBands
        ])

        orbit_filter = ee.Filter.eq("orbitProperties_pass", "ASCENDING" if self.ascending else "DESCENDING")

        s1_img = (
            ee.ImageCollection(self.imageCollectionName)
            .filter(ee_filter)
            .filter(polarization_filter)
            .filter(orbit_filter)
        )

        if self.use_edge_mask:
            s1_img = s1_img.map(self._mask_edge)

        s1_img = s1_img.select(list(self.availableBands.values()), list(self.availableBands.keys()))

        if self.selectedIndices:
            s1_img = s1_img.map(
                partial(ee_add_indexes_to_image, indexes=[expression for (expression, _, _) in self.selectedIndices])
            )

        s1_img = s1_img.select(
            [natural_band_name for natural_band_name, _ in self.selectedBands]
            + [indice_name for _, indice_name, _ in self.selectedIndices],
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

        s1_img = ee_filter_img_collection_invalid_pixels(s1_img, ee_geometry, self.pixelSize, self.minValidPixelCount)

        return ee.ImageCollection(s1_img)

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

        s1_img = self.imageCollection(ee_feature)

        features = s1_img.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )

        return features
