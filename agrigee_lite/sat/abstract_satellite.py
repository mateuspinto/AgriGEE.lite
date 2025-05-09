import ee


class AbstractSatellite:
    def __init__(self) -> None:
        self.startDate = ""
        self.endDate = ""
        self.shortName = "IDoNotExist"
        self.originalBands: list[str] = []
        self.renamed_bands: list[str] = []
        self.selectedBands: dict[str, str] = {}
        self.imageCollectionName = ""
        self.scaleBands = lambda x: x
        self.pixelSize: int = 0

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        return ee.ImageCollection()

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: list[str] | None = None,
        date_types: list[str] | None = None,
    ) -> ee.FeatureCollection:
        return ee.FeatureCollection()


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
