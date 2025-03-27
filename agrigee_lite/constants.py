bands_order = {
    0: "indexnum",
    1: "doy",
    11: "blue",
    12: "green",
    13: "red",
    14: "re1",
    15: "re2",
    16: "re3",
    17: "nir",
    18: "re4",
    19: "swir1",
    20: "swir2",
}

ALL_BANDS = {v: f"{k:02}_{v}" for k, v in bands_order.items()}
