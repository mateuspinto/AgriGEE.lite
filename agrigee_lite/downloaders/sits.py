import ee
import pandas as pd
from shapely import Polygon

from agrigee_lite.satellites.abstract_satellite import AbstractSatellite


def download_single_sits(
    geometry: Polygon, start_date: pd.Timestamp | str, end_date: pd.Timestamp | str, satellite: AbstractSatellite
) -> pd.DataFrame:
    start_date = start_date.strftime("%Y-%m-%d") if isinstance(start_date, pd.Timestamp) else start_date
    end_date = end_date.strftime("%Y-%m-%d") if isinstance(end_date, pd.Timestamp) else end_date

    ee_feature = ee.Feature(
        ee.Geometry(geometry.__geo_interface__),
        {"start_date": start_date, "end_date": end_date, "index_num": 1},
    )
    ee_expression = satellite.compute(ee_feature)
    return ee.data.computeFeatures({"expression": ee_expression, "fileFormat": "PANDAS_DATAFRAME"}).drop(
        columns=["geo", "index_num"]
    )
