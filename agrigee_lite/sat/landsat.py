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
    _DEFAULT_BANDS: list[str] = [  # noqa: RUF012
        "blue",
        "green",
        "red",
        "nir",
        "swir1",
        "swir2",
    ]

    def __init__(
        self,
        *,
        sensor_code: str,  # e.g. "LT05"
        toa_band_map: dict[str, str],
        sr_band_map: dict[str, str],
        short_base: str,  # e.g. "l5"
        start_date: str,  # sensor-specific
        end_date: str,  # sensor-specific
        bands: list[str] | None = None,
        indices: list[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
    ) -> None:
        super().__init__()

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

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        geom = ee_feature.geometry()
        ee_filter = ee.Filter.And(
            ee.Filter.bounds(geom),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        col = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)
        col = col.map(ee_l_apply_sr_scale_factors) if self.useSr else col.map(remove_l_toa_tough_clouds)

        col = col.select(list(self.availableBands.values()), list(self.availableBands.keys()))
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

        col = ee_filter_img_collection_invalid_pixels(col, geom, self.pixelSize, 12)
        return ee.ImageCollection(col)

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
    ) -> ee.FeatureCollection:
        geom = ee_feature.geometry()
        geom = ee_safe_remove_borders(geom, self.pixelSize, 50000)
        ee_feature = ee_feature.setGeometry(geom)

        col = self.imageCollection(ee_feature)
        features = col.map(
            partial(
                ee_map_bands_and_doy,
                ee_feature=ee_feature,
                pixel_size=self.pixelSize,
                subsampling_max_pixels=ee_get_number_of_pixels(geom, subsampling_max_pixels, self.pixelSize),
                reducer=ee_get_reducers(reducers),
            )
        )
        return features


class Landsat5(AbstractLandsat):
    def __init__(
        self,
        bands: list[str] | None = None,
        indices: list[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
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
        )


class Landsat7(AbstractLandsat):
    def __init__(
        self,
        bands: list[str] | None = None,
        indices: list[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
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
        )


class Landsat8(AbstractLandsat):
    def __init__(
        self,
        bands: list[str] | None = None,
        indices: list[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
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
        )


class Landsat9(AbstractLandsat):
    def __init__(
        self,
        bands: list[str] | None = None,
        indices: list[str] | None = None,
        use_sr: bool = True,
        tier: int = 1,
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
        )
