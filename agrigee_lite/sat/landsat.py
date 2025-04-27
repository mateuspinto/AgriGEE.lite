from functools import partial

import ee

from agrigee_lite.ee_utils import (
    ee_filter_img_collection_invalid_pixels,
    ee_get_number_of_pixels,
    ee_get_reducers,
    ee_map_bands_and_doy,
)
from agrigee_lite.sat.abstract_satellite import AbstractSatellite


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
    thermal_bands = img.select("ST_B6").multiply(0.00341802).add(149.0)
    return img.addBands(optical_bands, None, True).addBands(thermal_bands, None, True)


class Landsat5(AbstractSatellite):
    def __init__(
        self,
        bands: list[str] | None = None,
        use_sr: bool = False,
        tier: int = 1,
    ):
        if bands is None:
            bands = ["blue", "green", "red", "nir", "swir1", "swir2"]

        super().__init__()
        self.useSr = use_sr
        self.tier = tier
        self.pixelSize: int = 30

        self.imageCollectionName = f"LANDSAT/LT05/C02/T{tier}_L2" if use_sr else f"LANDSAT/LT05/C02/T{tier}_TOA"
        self.startDate: str = "1984-03-01"
        self.endDate: str = "2013-05-05"
        self.shortName: str = "l5sr" if use_sr else "l5"

        if use_sr:
            self.availableBands = {
                "blue": "SR_B1",
                "green": "SR_B2",
                "red": "SR_B3",
                "nir": "SR_B4",
                "swir1": "SR_B5",
                "swir2": "SR_B7",
            }
        else:
            self.availableBands = {
                "blue": "B1",
                "green": "B2",
                "red": "B3",
                "nir": "B4",
                "swir1": "B5",
                "swir2": "B7",
            }

        remap_bands = {name: f"{idx}_{name}" for idx, name in enumerate(bands)}

        self.selectedBands: dict[str, str] = {
            remap_bands[b]: self.availableBands[b] for b in bands if b in self.availableBands
        }
        self.selectedBands["cloudq"] = "QA_PIXEL"
        self.scaleBands = lambda x: x

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_filter = ee.Filter.And(
            ee.Filter.bounds(ee_geometry),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        l_img = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        l_img = l_img.map(ee_l_apply_sr_scale_factors) if self.useSr else l_img.map(remove_l_toa_tough_clouds)

        l_img = l_img.select(
            list(self.selectedBands.values()),
            list(self.selectedBands.keys()),
        )

        l_img = l_img.map(ee_l_mask)
        l_img = ee_filter_img_collection_invalid_pixels(l_img, ee_geometry, self.pixelSize, 12)

        return ee.ImageCollection(l_img)

    def compute(
        self,
        ee_feature: ee.Feature,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
        subsampling_max_pixels: float = 1000,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee.Geometry(
            ee.Algorithms.If(
                ee_geometry.buffer(-self.pixelSize).area().gte(50000),
                ee_geometry.buffer(-self.pixelSize),
                ee_geometry,
            )
        )

        l_img = self.imageCollection(ee_feature)

        features = l_img.map(
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

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class Landsat7(AbstractSatellite):
    def __init__(
        self,
        bands: list[str] | None = None,
        use_sr: bool = False,
        tier: int = 1,
    ):
        if bands is None:
            bands = ["blue", "green", "red", "nir", "swir1", "swir2"]

        super().__init__()
        self.useSr = use_sr
        self.tier = tier
        self.pixelSize: int = 30

        self.imageCollectionName = f"LANDSAT/LE07/C02/T{tier}_L2" if use_sr else f"LANDSAT/LE07/C02/T{tier}_TOA"
        self.startDate: str = "1984-03-01"
        self.endDate: str = "2013-05-05"
        self.shortName: str = "l7sr" if use_sr else "l7"

        if use_sr:
            self.availableBands = {
                "blue": "SR_B1",
                "green": "SR_B2",
                "red": "SR_B3",
                "nir": "SR_B4",
                "swir1": "SR_B5",
                "swir2": "SR_B7",
            }
        else:
            self.availableBands = {
                "blue": "B1",
                "green": "B2",
                "red": "B3",
                "nir": "B4",
                "swir1": "B5",
                "swir2": "B7",
            }

        remap_bands = {name: f"{idx}_{name}" for idx, name in enumerate(bands)}

        self.selectedBands: dict[str, str] = {
            remap_bands[b]: self.availableBands[b] for b in bands if b in self.availableBands
        }
        self.selectedBands["cloudq"] = "QA_PIXEL"
        self.scaleBands = lambda x: x

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_filter = ee.Filter.And(
            ee.Filter.bounds(ee_geometry),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        l_img = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        l_img = l_img.map(ee_l_apply_sr_scale_factors) if self.useSr else l_img.map(remove_l_toa_tough_clouds)

        l_img = l_img.select(
            list(self.selectedBands.values()),
            list(self.selectedBands.keys()),
        )

        l_img = l_img.map(ee_l_mask)
        l_img = ee_filter_img_collection_invalid_pixels(l_img, ee_geometry, self.pixelSize, 12)

        return ee.ImageCollection(l_img)

    def compute(
        self,
        ee_feature: ee.Feature,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
        subsampling_max_pixels: float = 1000,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee.Geometry(
            ee.Algorithms.If(
                ee_geometry.buffer(-self.pixelSize).area().gte(50000),
                ee_geometry.buffer(-self.pixelSize),
                ee_geometry,
            )
        )

        l_img = self.imageCollection(ee_feature)

        features = l_img.map(
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

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class Landsat8(AbstractSatellite):
    def __init__(
        self,
        bands: list[str] | None = None,
        use_sr: bool = False,
        tier: int = 1,
    ):
        if bands is None:
            bands = ["blue", "green", "red", "nir", "swir1", "swir2"]

        super().__init__()
        self.useSr = use_sr
        self.tier = tier
        self.pixelSize: int = 30

        self.imageCollectionName = f"LANDSAT/LC08/C02/T{tier}_L2" if use_sr else f"LANDSAT/LC08/C02/T{tier}_TOA"
        self.startDate: str = "1984-03-01"
        self.endDate: str = "2013-05-05"
        self.shortName: str = "l8sr" if use_sr else "l8"

        if use_sr:
            self.availableBands = {
                "blue": "SR_B2",
                "green": "SR_B3",
                "red": "SR_B4",
                "nir": "SR_B5",
                "swir1": "SR_B6",
                "swir2": "SR_B7",
            }
        else:
            self.availableBands = {
                "blue": "B2",
                "green": "B3",
                "red": "B4",
                "nir": "B5",
                "swir1": "B6",
                "swir2": "B7",
            }

        remap_bands = {name: f"{idx}_{name}" for idx, name in enumerate(bands)}

        self.selectedBands: dict[str, str] = {
            remap_bands[b]: self.availableBands[b] for b in bands if b in self.availableBands
        }
        self.selectedBands["cloudq"] = "QA_PIXEL"
        self.scaleBands = lambda x: x

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_filter = ee.Filter.And(
            ee.Filter.bounds(ee_geometry),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        l_img = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        l_img = l_img.map(ee_l_apply_sr_scale_factors) if self.useSr else l_img.map(remove_l_toa_tough_clouds)

        l_img = l_img.select(
            list(self.selectedBands.values()),
            list(self.selectedBands.keys()),
        )

        l_img = l_img.map(ee_l_mask)
        l_img = ee_filter_img_collection_invalid_pixels(l_img, ee_geometry, self.pixelSize, 12)

        return ee.ImageCollection(l_img)

    def compute(
        self,
        ee_feature: ee.Feature,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
        subsampling_max_pixels: float = 1000,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee.Geometry(
            ee.Algorithms.If(
                ee_geometry.buffer(-self.pixelSize).area().gte(50000),
                ee_geometry.buffer(-self.pixelSize),
                ee_geometry,
            )
        )

        l_img = self.imageCollection(ee_feature)

        features = l_img.map(
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

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class Landsat9(AbstractSatellite):
    def __init__(
        self,
        bands: list[str] | None = None,
        use_sr: bool = False,
        tier: int = 1,
    ):
        if bands is None:
            bands = ["blue", "green", "red", "nir", "swir1", "swir2"]

        super().__init__()
        self.useSr = use_sr
        self.tier = tier
        self.pixelSize: int = 30

        self.imageCollectionName = f"LANDSAT/LC09/C02/T{tier}_L2" if use_sr else f"LANDSAT/LC09/C02/T{tier}_TOA"
        self.startDate: str = "1984-03-01"
        self.endDate: str = "2013-05-05"
        self.shortName: str = "l9sr" if use_sr else "l9"

        if use_sr:
            self.availableBands = {
                "blue": "SR_B2",
                "green": "SR_B3",
                "red": "SR_B4",
                "nir": "SR_B5",
                "swir1": "SR_B6",
                "swir2": "SR_B7",
            }
        else:
            self.availableBands = {
                "blue": "B2",
                "green": "B3",
                "red": "B4",
                "nir": "B5",
                "swir1": "B6",
                "swir2": "B7",
            }

        remap_bands = {name: f"{idx}_{name}" for idx, name in enumerate(bands)}

        self.selectedBands: dict[str, str] = {
            remap_bands[b]: self.availableBands[b] for b in bands if b in self.availableBands
        }
        self.selectedBands["cloudq"] = "QA_PIXEL"
        self.scaleBands = lambda x: x

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        ee_geometry = ee_feature.geometry()
        ee_filter = ee.Filter.And(
            ee.Filter.bounds(ee_geometry),
            ee.Filter.date(ee_feature.get("s"), ee_feature.get("e")),
        )

        l_img = ee.ImageCollection(self.imageCollectionName).filter(ee_filter)

        l_img = l_img.map(ee_l_apply_sr_scale_factors) if self.useSr else l_img.map(remove_l_toa_tough_clouds)

        l_img = l_img.select(
            list(self.selectedBands.values()),
            list(self.selectedBands.keys()),
        )

        l_img = l_img.map(ee_l_mask)
        l_img = ee_filter_img_collection_invalid_pixels(l_img, ee_geometry, self.pixelSize, 12)

        return ee.ImageCollection(l_img)

    def compute(
        self,
        ee_feature: ee.Feature,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
        subsampling_max_pixels: float = 1000,
    ) -> ee.FeatureCollection:
        ee_geometry = ee_feature.geometry()
        ee_geometry = ee.Geometry(
            ee.Algorithms.If(
                ee_geometry.buffer(-self.pixelSize).area().gte(50000),
                ee_geometry.buffer(-self.pixelSize),
                ee_geometry,
            )
        )

        l_img = self.imageCollection(ee_feature)

        features = l_img.map(
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

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName
