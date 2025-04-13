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

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        return ee.ImageCollection()

    def compute(self, ee_feature: ee.Feature) -> ee.FeatureCollection:
        return ee.FeatureCollection()
