import ee


class AbstractSatellite:
    def __init__(self) -> None:
        self.startDate = ""
        self.endDate = ""
        self.originalBands: list[str] = []
        self.renamed_bands: list[str] = []
        self.imageCollectionName = ""

    def compute(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        return ee.ImageCollection()
