from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
    ee_safe_remove_borders,
)
from agrigee_lite.sat.abstract_satellite import RadarSatellite


class Sentinel1(RadarSatellite):
    """
    Satellite abstraction for Sentinel-1.

    Sentinel-1 is a constellation of two Earth observation satellites,
    operated by ESA, designed for radar imaging of land and ocean surfaces. Sentinel-1B suffered a power failure in December 2021 and is no longer operational.

    Parameters
    ----------
    polarization : list of str, optional
        List of polarizations to select. Defaults to 'VV' and 'VH' for land applications.
    orbit_pass : str, optional
        Preferred orbit pass ('ASCENDING' or 'DESCENDING'). Defaults to ASCENDING.

    Satellite Information
    ---------------------
    +-----------------------------------+------------------------+
    | Field                             | Value                  |
    +-----------------------------------+------------------------+
    | Name                              | Sentinel-1             |
    | Revisit Time                      | 6 days (constellation) |
    | Revisit Time (after 2021)          | 12 days (only S1A)    |
    | Pixel Size                        | ~10 meters             |
    | Coverage                          | Global                 |
    +-----------------------------------+------------------------+

    Collection Dates
    ----------------
    +------------------+------------+----------------+
    | Collection Type  | Start Date | End Date        |
    +------------------+------------+----------------+
    | GRD (Ground Range Detected) | 2014-10-03 | present   |
    +------------------+------------+----------------+

    Band Information
    ------------
    +------------+---------------+--------------+----------------------+
    | Band Name  | Original Band | Resolution   | Spectral Wavelength   |
    +------------+---------------+--------------+----------------------+
    | VV         | C-band         | ~10 m        | 5.405 GHz (~5.5 cm)   |
    | VH         | C-band         | ~10 m        | 5.405 GHz (~5.5 cm)   |
    +------------+---------------+--------------+----------------------+
    """

    def __init__(
        self,
        bands: list[str] | None = None,
        ascending: bool = True,
    ):
        if bands is None:
            bands = ["vv", "vh"]

        super().__init__()

        self.ascending: bool = ascending
        self.imageCollectionName: str = "COPERNICUS/S1_GRD"
        self.pixelSize: int = 10

        # full mission start (S-1A launch)
        self.startDate: str = "2014-10-03"
        self.endDate: str = "2050-01-01"
        self.shortName: str = "s1a" if ascending else "s1d"

        # original → product band
        self.availableBands: dict[str, str] = {"vv": "VV", "vh": "VH"}

        remap_bands = {s: f"{n}_{s}" for n, s in enumerate(bands)}
        self.selectedBands: dict[str, str] = {
            remap_bands[b]: self.availableBands[b] for b in bands if b in self.availableBands
        }

        # keep values in dB - no scaling needed
        self.scaleBands = lambda x: x

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
        ee_geometry = ee_safe_remove_borders(ee_geometry, self.pixelSize, 35000)

        ee_filter = ee.Filter.And(ee.Filter.bounds(ee_geometry), ee.Filter.date(ee_start, ee_end))

        s1_img = (
            ee.ImageCollection(self.imageCollectionName)
            .filter(ee_filter)
            .filter(
                ee.Filter.And(
                    ee.Filter.listContains("transmitterReceiverPolarisation", "VV"),
                    ee.Filter.listContains("transmitterReceiverPolarisation", "VH"),
                )
            )
            .filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING" if self.ascending else "DESCENDING"))
            .map(self._mask_edge)
            .select(list(self.availableBands.values()), list(self.selectedBands.keys()))
        )

        s1_img = ee_filter_img_collection_invalid_pixels(s1_img, ee_geometry, self.pixelSize, 20)

        return ee.ImageCollection(s1_img)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee.Geometry(
            ee.Algorithms.If(ee_geometry.buffer(-10).area().gte(35000), ee_geometry.buffer(-10), ee_geometry)
        )

        s1_img = self.imageCollection(ee_feature)

        features = s1_img.map(
            partial(
                ee_map_bands_and_doy,
                ee_geometry=ee_geometry,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(ee_geometry, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
                date_types=date_types,
            )
        )

        return features

    # --------------------------------------------------------------------- #

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
