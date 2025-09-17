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
    """
    ⚠️⚠️⚠️ Sentinel-1 Availability Warning
    ---------------------------------
    Due to the failure of the Sentinel-1B satellite in December 2021, the constellation has been operating solely
    with Sentinel-1A. This has led to reduced data availability in many regions — particularly in the Southern
    Hemisphere — with revisit times increasing from ~6 days to ~12 days or more. Some areas may experience
    significant temporal gaps, especially after early 2022. ⚠️⚠️⚠️

    Satellite abstraction for Sentinel-1 Ground Range Detected (GRD) product.

    Sentinel-1 is a constellation of two polar-orbiting satellites (Sentinel-1A and 1B)
    operated by ESA, equipped with C-band Synthetic Aperture Radar (SAR). It provides
    all-weather, day-and-night imaging of Earth's surface.

    This class wraps the Sentinel-1 GRD product and allows users to select polarizations,
    filter by orbit pass, and apply edge masks to remove low-backscatter areas (e.g., layover).

    Parameters
    ----------
    bands : set of str, optional
        Set of polarizations to select. Defaults to {'vv', 'vh'}.
    indices : set of str, optional
        Set of radar indices (e.g. ratios). Defaults to [].
    ascending : bool, default=True
        If True, selects ASCENDING orbit passes. If False, selects DESCENDING.
    use_edge_mask : bool, optional
        Whether to apply an edge mask to remove extreme low-backscatter areas
        (commonly occurring near the edges of acquisitions or in layover/shadow zones).
        Default is True.
    min_valid_pixel_count : int, default=20
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=35_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Edge Masking
    ------------
    Sentinel-1 radar images often contain low-backscatter areas near image borders or over layover zones.
    This class applies a threshold-based edge mask (`< -30 dB`) to reduce artifacts.

    Satellite Information
    ---------------------
    +-------------------------------+-------------------------------+
    | Field                         | Value                         |
    +-------------------------------+-------------------------------+
    | Name                          | Sentinel-1                    |
    | Agency                        | ESA (Copernicus)              |
    | Instrument                    | C-band Synthetic Aperture Radar (SAR) |
    | Revisit Time (full mission)   | ~6 days (1A + 1B constellation)|
    | Revisit Time (post-2021)      | ~12 days (only 1A active)     |
    | Orbit Type                    | Sun-synchronous (polar)       |
    | Pixel Size                    | ~10 meters                    |
    | Coverage                      | Global                        |
    +-------------------------------+-------------------------------+

    Collection Dates
    ----------------
    +------------------+-------------+-----------+
    | Product          | Start Date  | End Date  |
    +------------------+-------------+-----------+
    | GRD              | 2014-10-03  | present   |
    +------------------+-------------+-----------+

    Band Information
    ----------------
    +------------+-----------+-------------+------------------------------+
    | Band Name  | Frequency | Resolution  | Description                  |
    +------------+-----------+-------------+------------------------------+
    | VV         | 5.405 GHz | ~10 meters  | Vertical transmit/receive    |
    | VH         | 5.405 GHz | ~10 meters  | Vertical transmit, horizontal receive |
    +------------+-----------+-------------+------------------------------+

    Notes
    -----
    - Official GRD collection (Earth Engine):
      https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD

    - Sentinel-1 User Guide:
      https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-1-sar

    - Orbit direction filter:
      https://developers.google.com/earth-engine/sentinel1#orbit-direction
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
        bands = sorted({"vv", "vh"}) if bands is None else sorted(bands)

        indices = [] if indices is None else sorted(indices)

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

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
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

        edge = img.lt(-30.0)
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
