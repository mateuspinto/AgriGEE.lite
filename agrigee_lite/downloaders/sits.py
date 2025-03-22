import ee
import pandas as pd
from shapely import Polygon

from agrigee_lite.satellites.abstract_satellite import AbstractSatellite


def download_single(
    geometry: Polygon, start_date: pd.Timestamp, end_date: pd.Timestamp, satellite: AbstractSatellite
) -> pd.DataFrame:
    ee_feature = ee.Feature(
        ee.Geometry(geometry.__geo_interface__),
        {"start_date": start_date.strftime("%Y-%m-%d"), "end_date": end_date.strftime("%Y-%m-%d"), "index_num": 1},
    )
    ee_expression = satellite.compute(ee_feature)
    return ee.data.computeFeatures({"expression": ee_expression, "fileFormat": "PANDAS_DATAFRAME"}).drop(
        columns=["geo", "index_num"]
    )
