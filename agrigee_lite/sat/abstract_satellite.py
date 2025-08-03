import ee

from agrigee_lite.vegetation_indices import VEGETATION_INDICES


class AbstractSatellite:
    def __init__(self) -> None:
        self.startDate = ""
        self.endDate = ""
        self.shortName = "IDoNotExist"
        self.availableBands: dict[str, str] = {}
        self.selectedBands: list[tuple[str, str]] = []
        self.selectedIndices: list[str] = []
        self.imageCollectionName = ""
        self.pixelSize: int = 0

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        return ee.ImageCollection()

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
    ) -> ee.FeatureCollection:
        return ee.FeatureCollection()

    def log_dict(self) -> dict:
        return {self.__class__.__name__: self.__dict__}

    @property
    def availableIndices(self) -> dict[str, str]:
        return {
            name: idx["expression"]
            for name, idx in VEGETATION_INDICES.items()
            if idx["required_bands"].issubset(self.availableBands.keys())
        }

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class OpticalSatellite(AbstractSatellite):
    def __init__(self) -> None:
        super().__init__()
        self.dateType = "optical"


class RadarSatellite(AbstractSatellite):
    def __init__(self) -> None:
        super().__init__()
        self.dateType = "radar"


class DataSourceSatellite(AbstractSatellite):
    def __init__(self) -> None:
        super().__init__()
        self.dateType = "dataSource"


class SingleImageSatellite(AbstractSatellite):
    def __init__(self) -> None:
        super().__init__()
        self.dateType = "singleImage"
