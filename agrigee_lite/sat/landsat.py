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


def remove_l_toa_tough_clouds(img: ee.Image) -> ee.Image:
    img = ee.Image(img)
    img = ee.Algorithms.Landsat.simpleCloudScore(img)

    mask = img.select(["cloud"]).lte(15)
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
    _DEFAULT_BANDS: set[str] = {  # noqa: RUF012
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
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
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ) -> None:
        super().__init__()

        bands = sorted(self._DEFAULT_BANDS) if bands is None else sorted(bands)

        indices = [] if indices is None else sorted(indices)

        if indices is None:
            indices = []

        bands = bands or self._DEFAULT_BANDS
        self.useSr = use_sr
        self.tier = tier
        self.pixelSize: int = 30

        self.startDate: str = start_date
        self.endDate: str = end_date

        suffix = "L2" if use_sr else "TOA"
        self.imageCollectionName = f"LANDSAT/{sensor_code}/C02/T{tier}_{suffix}"
        self.shortName: str = f"{short_base}sr" if use_sr else short_base

        self.availableBands = sr_band_map if use_sr else toa_band_map
        self.availableBands["cloudq"] = "QA_PIXEL"

        self.selectedBands: list[tuple[str, str]] = [(band, f"{(n + 10):02}_{band}") for n, band in enumerate(bands)]

        self.selectedIndices: list[str] = [
            (self.availableIndices[indice_name], indice_name, f"{(n + 40):02}_{indice_name}")
            for n, indice_name in enumerate(indices)
        ]

        self.useCloudMask = use_cloud_mask
        self.minValidPixelCount = min_valid_pixel_count
        self.minAreaToKeepBorder = min_area_to_keep_border
        self.borderPixelsToErode = border_pixels_to_erode

        self.toDownloadSelectors = (
            [numeral_band_name for _, numeral_band_name in self.selectedBands]
            + [numeral_indice_name for _, _, numeral_indice_name in self.selectedIndices],
        )

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
            col = col.map(remove_l_toa_tough_clouds)

        col = col.select(list(self.availableBands.values()), list(self.availableBands.keys()))

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
    """
    Satellite abstraction for Landsat 5 (TM sensor, Collection 2).

    Landsat 5 was launched in 1984 and provided more than 29 years of Earth observation data.
    This class supports both TOA and SR products, with optional cloud masking using the QA_PIXEL band.

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'].
    indices : set of str, optional
        Spectral indices to compute from the selected bands.
    use_sr : bool, default=True
        Whether to use surface reflectance products ('SR_B*' bands).
        If False, uses top-of-atmosphere reflectance ('B*' bands).
    tier : int, default=1
        Landsat collection tier to use (1 or 2). Tier 1 has highest geometric accuracy.
    use_cloud_mask : bool, default=True
        Whether to apply QA_PIXEL-based cloud masking. If False, no cloud mask is applied.
    min_valid_pixel_count : int, default=12
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=50_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Cloud Masking
    -------------
    Cloud masking is based on the QA_PIXEL band, using bit flags defined by USGS:
    - Applied to both TOA and SR products when `use_cloud_mask=True`
    - For TOA collections, an additional filter (`remove_l_toa_tough_clouds`) is applied
    to remove low-quality observations based on a simple cloud scoring method.

    Satellite Information
    ---------------------
    +----------------------------+------------------------+
    | Field                      | Value                  |
    +----------------------------+------------------------+
    | Name                       | Landsat 5 TM           |
    | Sensor                     | TM (Thematic Mapper)   |
    | Platform                   | Landsat 5              |
    | Temporal Resolution        | 16 days                |
    | Pixel Size                 | 30 meters              |
    | Coverage                   | Global                 |
    +----------------------------+------------------------+

    Collection Dates
    ----------------
    +-------------+------------+------------+
    | Product     | Start Date | End Date  |
    +-------------+------------+------------+
    | TOA         | 1984-03-01 | 2013-05-05 |
    | SR          | 1984-03-01 | 2012-05-05 |
    +-------------+------------+------------+

    Band Information
    ----------------
    +-----------+----------+-----------+------------------------+
    | Band Name | TOA Name | SR Name   | Spectral Wavelength    |
    +-----------+----------+-----------+------------------------+
    | blue      | B1       | SR_B1     | 450-520 nm             |
    | green     | B2       | SR_B2     | 520-600 nm             |
    | red       | B3       | SR_B3     | 630-690 nm             |
    | nir       | B4       | SR_B4     | 770-900 nm             |
    | swir1     | B5       | SR_B5     | 1550-1750 nm           |
    | swir2     | B7       | SR_B7     | 2090-2350 nm           |
    +-----------+----------+-----------+------------------------+

    Notes
    -----
    - Landsat 5 TOA Collection (Tier 1):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_TOA

    - Landsat 5 TOA Collection (Tier 2):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T2_TOA

    - Landsat 5 SR Collection (Tier 1):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2

    - Landsat 5 SR Collection (Tier 2):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T2_L2

    - Cloud mask reference (QA_PIXEL flags):
        https://www.usgs.gov/media/files/landsat-collection-2-pixel-quality-assessment

    - TOA cloud filtering (Simple Cloud Score):
        https://developers.google.com/earth-engine/guides/landsat?hl=pt-br#simple-cloud-score
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
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
            border_pixels_to_erode=border_pixels_to_erode,
            min_area_to_keep_border=min_area_to_keep_border,
        )


class Landsat7(AbstractLandsat):
    """
    Satellite abstraction for Landsat 7 (ETM+ sensor, Collection 2).

    Landsat 7 was launched in 1999 and provided over two decades of data.
    This class supports both TOA and SR products, with optional cloud masking using the QA_PIXEL band.

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'].
    indices : set of str, optional
        Spectral indices to compute from the selected bands.
    use_sr : bool, default=True
        Whether to use surface reflectance products ('SR_B*' bands).
        If False, uses top-of-atmosphere reflectance ('B*' bands).
    tier : int, default=1
        Landsat collection tier to use (1 or 2). Tier 1 has highest geometric accuracy.
    use_cloud_mask : bool, default=True
        Whether to apply QA_PIXEL-based cloud masking. If False, no cloud mask is applied.
    min_valid_pixel_count : int, default=12
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=50_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Cloud Masking
    -------------
    Cloud masking is based on the QA_PIXEL band, using bit flags defined by USGS:
    - Applied to both TOA and SR products when `use_cloud_mask=True`
    - For TOA collections, an additional filter (`remove_l_toa_tough_clouds`) is applied
    to remove low-quality observations based on a simple cloud scoring method.

    Satellite Information
    ---------------------
    +----------------------------+------------------------+
    | Field                      | Value                  |
    +----------------------------+------------------------+
    | Name                       | Landsat 7 ETM+         |
    | Sensor                     | ETM+ (Enhanced TM Plus)|
    | Platform                   | Landsat 7              |
    | Temporal Resolution        | 16 days                |
    | Pixel Size                 | 30 meters              |
    | Coverage                   | Global                 |
    +----------------------------+------------------------+

    Collection Dates
    ----------------
    +-------------+------------+------------+
    | Product     | Start Date | End Date  |
    +-------------+------------+------------+
    | TOA         | 1999-04-15 | 2022-04-06 |
    | SR          | 1999-04-15 | 2022-04-06 |
    +-------------+------------+------------+

    Band Information
    ----------------
    +-----------+----------+-----------+------------------------+
    | Band Name | TOA Name | SR Name   | Spectral Wavelength    |
    +-----------+----------+-----------+------------------------+
    | blue      | B1       | SR_B1     | 450-520 nm             |
    | green     | B2       | SR_B2     | 520-600 nm             |
    | red       | B3       | SR_B3     | 630-690 nm             |
    | nir       | B4       | SR_B4     | 770-900 nm             |
    | swir1     | B5       | SR_B5     | 1550-1750 nm           |
    | swir2     | B7       | SR_B7     | 2090-2350 nm           |
    +-----------+----------+-----------+------------------------+

    Notes
    -----
    - Landsat 7 TOA Collection (Tier 1):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_TOA

    - Landsat 7 TOA Collection (Tier 2):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T2_TOA

    - Landsat 7 SR Collection (Tier 1):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2

    - Landsat 7 SR Collection (Tier 2):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T2_L2

    - Cloud mask reference (QA_PIXEL flags):
        https://www.usgs.gov/media/files/landsat-collection-2-pixel-quality-assessment

    - TOA cloud filtering (Simple Cloud Score):
        https://developers.google.com/earth-engine/guides/landsat?hl=pt-br#simple-cloud-score
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
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
            border_pixels_to_erode=border_pixels_to_erode,
            min_area_to_keep_border=min_area_to_keep_border,
        )


class Landsat8(AbstractLandsat):
    """
    Satellite abstraction for Landsat 8 (OLI/TIRS sensor, Collection 2).

    Landsat 8 was launched in 2013 and remains in operation, delivering high-quality Earth observation data.
    This class supports both TOA and SR products, with optional cloud masking using the QA_PIXEL band.

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'].
    indices : set of str, optional
        Spectral indices to compute from the selected bands.
    use_sr : bool, default=True
        Whether to use surface reflectance products ('SR_B*' bands).
        If False, uses top-of-atmosphere reflectance ('B*' bands).
    tier : int, default=1
        Landsat collection tier to use (1 or 2). Tier 1 has highest geometric accuracy.
    use_cloud_mask : bool, default=True
        Whether to apply QA_PIXEL-based cloud masking. If False, no cloud mask is applied.
    min_valid_pixel_count : int, default=12
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=50_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Cloud Masking
    -------------
    Cloud masking is based on the QA_PIXEL band, using bit flags defined by USGS:
    - Applied to both TOA and SR products when `use_cloud_mask=True`
    - For TOA collections, an additional filter (`remove_l_toa_tough_clouds`) is applied
    to remove low-quality observations based on a simple cloud scoring method.

    Satellite Information
    ---------------------
    +----------------------------+------------------------+
    | Field                      | Value                  |
    +----------------------------+------------------------+
    | Name                       | Landsat 8 OLI/TIRS     |
    | Sensor                     | OLI + TIRS             |
    | Platform                   | Landsat 8              |
    | Temporal Resolution        | 16 days                |
    | Pixel Size                 | 30 meters              |
    | Coverage                   | Global                 |
    +----------------------------+------------------------+

    Collection Dates
    ----------------
    +-------------+------------+------------+
    | Product     | Start Date | End Date  |
    +-------------+------------+------------+
    | TOA         | 2013-04-11 | present   |
    | SR          | 2013-04-11 | present   |
    +-------------+------------+------------+

    Band Information
    ----------------
    +-----------+----------+-----------+------------------------+
    | Band Name | TOA Name | SR Name   | Spectral Wavelength    |
    +-----------+----------+-----------+------------------------+
    | blue      | B2       | SR_B2     | 450-515 nm             |
    | green     | B3       | SR_B3     | 525-600 nm             |
    | red       | B4       | SR_B4     | 630-680 nm             |
    | nir       | B5       | SR_B5     | 845-885 nm             |
    | swir1     | B6       | SR_B6     | 1560-1660 nm           |
    | swir2     | B7       | SR_B7     | 2100-2300 nm           |
    +-----------+----------+-----------+------------------------+

    Notes
    -----
    - Landsat 8 TOA Collection (Tier 1):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_TOA

    - Landsat 8 TOA Collection (Tier 2):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T2_TOA

    - Landsat 8 SR Collection (Tier 1):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2

    - Landsat 8 SR Collection (Tier 2):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T2_L2

    - Cloud mask reference (QA_PIXEL flags):
        https://www.usgs.gov/media/files/landsat-collection-2-pixel-quality-assessment

    - TOA cloud filtering (Simple Cloud Score):
        https://developers.google.com/earth-engine/guides/landsat?hl=pt-br#simple-cloud-score
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ):
        toa = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6", "swir2": "B7"}
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
            border_pixels_to_erode=border_pixels_to_erode,
            min_area_to_keep_border=min_area_to_keep_border,
        )


class Landsat9(AbstractLandsat):
    """
    Satellite abstraction for Landsat 9 (OLI-2/TIRS-2 sensor, Collection 2).

    Landsat 9 is the latest mission in the Landsat program, launched in 2021. It is nearly identical to Landsat 8
    and provides continuity for high-quality multispectral Earth observation. This class supports both TOA and SR
    products, with optional cloud masking using the QA_PIXEL band.

    Parameters
    ----------
    bands : set of str, optional
        Set of bands to select. Defaults to ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'].
    indices : set of str, optional
        Spectral indices to compute from the selected bands.
    use_sr : bool, default=True
        Whether to use surface reflectance products ('SR_B*' bands).
        If False, uses top-of-atmosphere reflectance ('B*' bands).
    tier : int, default=1
        Landsat collection tier to use (1 or 2). Tier 1 has highest geometric accuracy.
    use_cloud_mask : bool, default=True
        Whether to apply QA_PIXEL-based cloud masking. If False, no cloud mask is applied.
    min_valid_pixel_count : int, default=12
        Minimum number of valid (non-cloud) pixels required to retain an image.
    border_pixels_to_erode : float, default=1
        Number of pixels to erode from the geometry border.
    min_area_to_keep_border : int, default=50_000
        Minimum area (in m²) required to retain geometry after border erosion.

    Cloud Masking
    -------------
    Cloud masking is based on the QA_PIXEL band, using bit flags defined by USGS:
    - Applied to both TOA and SR products when `use_cloud_mask=True`
    - For TOA collections, an additional filter (`remove_l_toa_tough_clouds`) is applied
    to remove low-quality observations based on a simple cloud scoring method.

    Satellite Information
    ---------------------
    +----------------------------+------------------------+
    | Field                      | Value                  |
    +----------------------------+------------------------+
    | Name                       | Landsat 9 OLI-2/TIRS-2 |
    | Sensor                     | OLI-2 + TIRS-2         |
    | Platform                   | Landsat 9              |
    | Temporal Resolution        | 16 days                |
    | Pixel Size                 | 30 meters              |
    | Coverage                   | Global                 |
    +----------------------------+------------------------+

    Collection Dates
    ----------------
    +-------------+------------+------------+
    | Product     | Start Date | End Date  |
    +-------------+------------+------------+
    | TOA         | 2021-11-01 | present   |
    | SR          | 2021-11-01 | present   |
    +-------------+------------+------------+

    Band Information
    ----------------
    +-----------+----------+-----------+------------------------+
    | Band Name | TOA Name | SR Name   | Spectral Wavelength    |
    +-----------+----------+-----------+------------------------+
    | blue      | B2       | SR_B2     | 450-515 nm             |
    | green     | B3       | SR_B3     | 525-600 nm             |
    | red       | B4       | SR_B4     | 630-680 nm             |
    | nir       | B5       | SR_B5     | 845-885 nm             |
    | swir1     | B6       | SR_B6     | 1560-1660 nm           |
    | swir2     | B7       | SR_B7     | 2100-2300 nm           |
    +-----------+----------+-----------+------------------------+

    Notes
    -----
    - Landsat 9 TOA Collection (Tier 1):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_TOA

    - Landsat 9 TOA Collection (Tier 2):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T2_TOA

    - Landsat 9 SR Collection (Tier 1):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2

    - Landsat 9 SR Collection (Tier 2):
        https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T2_L2

    - Cloud mask reference (QA_PIXEL flags):
        https://www.usgs.gov/media/files/landsat-collection-2-pixel-quality-assessment

    - TOA cloud filtering (Simple Cloud Score):
        https://developers.google.com/earth-engine/guides/landsat?hl=pt-br#simple-cloud-score
    """

    def __init__(
        self,
        bands: set[str] | None = None,
        indices: set[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
        use_cloud_mask: bool = True,
        min_valid_pixel_count: int = 12,
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ):
        toa = {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6", "swir2": "B7"}
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
            border_pixels_to_erode=border_pixels_to_erode,
            min_area_to_keep_border=min_area_to_keep_border,
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
        border_pixels_to_erode: float = 1,
        min_area_to_keep_border: int = 50_000,
    ):
        raise NotImplementedError("HAHA FUNNY. Landsat 10 does not exist (yet).")
