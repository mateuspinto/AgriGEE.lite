import re

VEGETATION_INDICES_LIST = [
    "evi = 2.5 * ((i.nir - i.red)/(i.nir + 6 * i.red - 7.5 * i.blue + 1))",
    "evi2 = 2.5 * ((i.nir - i.red)/(i.nir + 2.4 * i.red + 1))",
    "ndvi = (i.nir - i.red)/(i.nir + i.red)",
    "ndwi = (i.nir - i.swir1)/(i.nir + i.swir1)",
    "mndwi = (i.green - i.swir1)/(i.green + i.swir1)",
    # Radar-specific indices
    "vhvv = (i.vv - i.vh)/(i.vv + i.vh)",  # VH-VV ratio (Sentinel-1)
    "hhhv = (i.hh - i.hv)/(i.hh + i.hv)",  # HH-HV ratio (PALSAR)
    "rvi = 4 * i.hv / (i.hh + i.hv)",  # Radar Vegetation Index (PALSAR)
    "ravi = 4 * i.vh / (i.vv + i.vh)",  # Radar Adapted Vegetation Index (Sentinel-1)
]

VEGETATION_INDICES = {}

for item in VEGETATION_INDICES_LIST:
    key, expression = item.split("=", 1)
    key = key.strip()
    expression = expression.strip()
    bands = set(re.findall(r"i\.([a-z0-9_]+)", expression))
    VEGETATION_INDICES[key] = {"expression": item, "required_bands": bands}
