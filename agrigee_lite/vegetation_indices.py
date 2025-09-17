import re

VEGETATION_INDICES_LIST = [
    "ndvi = (i.nir - i.red) / (i.nir + i.red)",
    "gndvi = (i.nir - i.green) / (i.nir + i.green)",
    "ndwi = (i.nir - i.swir1) / (i.nir + i.swir1)",
    "savi = ((i.nir - i.red) / (i.nir + i.red + 0.5)) * 1.5",
    "evi = 2.5 * (i.nir - i.red) / (i.nir + 6 * i.red - 7.5 * i.blue + 1)",
    "evi2 = 2.5 * (i.nir - i.red) / (i.nir + 2.4 * i.red + 1)",
    "msavi = (2 * i.nir + 1 - ((2 * i.nir + 1) ** 2 - 8 * (i.nir - i.red)) ** 0.5) / 2",
    "ndre = (i.nir - i.red_edge) / (i.nir + i.red_edge)",
    "mcari = ((i.nir - i.red) - 0.2 * (i.nir - i.green)) * (i.nir / i.red)",
    "gci = (i.nir / i.green) - 1",
    "bsi = ((i.swir1 + i.red) - (i.nir + i.blue)) / ((i.swir1 + i.red) + (i.nir + i.blue))",
    "ci_red = (i.nir / i.red) - 1",
    "ci_green = (i.nir / i.green) - 1",
    "osavi = (i.nir - i.red) / (i.nir + i.red + 0.16)",
    "arvi = (i.nir - (2 * i.red - i.blue)) / (i.nir + (2 * i.red - i.blue))",
    "mndwi = (i.green - i.swir1) / (i.green + i.swir1)",
    "hhhv = (i.hh - i.hv) / (i.hh + i.hv)",  # HH-HV ratio (PALSAR)
    "rvi = 4 * i.hv / (i.hh + i.hv)",  # Radar Vegetation Index (PALSAR)
    "vhvv = i.vh / i.vv",  # VH-VV ratio (Sentinel 1)
    "ravi = 4 * i.vh / (i.vv + i.vh)",  # Radar Adapted Vegetation Index (Sentinel-1)
]

VEGETATION_INDICES = {}

for item in VEGETATION_INDICES_LIST:
    key, expression = item.split("=", 1)
    key = key.strip()
    expression = expression.strip()
    bands = set(re.findall(r"i\.([a-z0-9_]+)", expression))
    VEGETATION_INDICES[key] = {"expression": item, "required_bands": bands}
