import json
import os
import random
import string

import ee
import geopandas as gpd
import numpy as np
import pandas as pd


def ee_map_bands_and_doy(
    ee_img: ee.Image,
    ee_feature: ee.Feature,
    pixel_size: int,
    subsampling_max_pixels: ee.Number,
    reducer: ee.Reducer,
    single_image: bool = False,
) -> ee.Feature:
    ee_img = ee.Image(ee_img)
    ee_stats = ee_img.reduceRegion(
        reducer=reducer,
        geometry=ee_feature.geometry(),
        scale=pixel_size,
        maxPixels=subsampling_max_pixels,
        bestEffort=True,
    ).map(lambda _, value: ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(value, None), 0, value)))

    if not single_image:
        ee_stats = ee_stats.set("01_timestamp", ee.Date(ee_img.date()).format("YYYY-MM-dd"))

    ee_stats = ee_stats.set("00_indexnum", ee_feature.get("0"))
    ee_stats = ee_stats.set("99_validPixelsCount", ee_img.get("ZZ_USER_VALID_PIXELS"))

    return ee.Feature(None, ee_stats)


def ee_map_valid_pixels(img: ee.Image, ee_geometry: ee.Geometry, pixel_size: int) -> ee.Image:
    mask = ee.Image(img).select([0]).gt(0)

    valid_pixels = ee.Number(
        mask.rename("valid")
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ee_geometry,
            scale=pixel_size,
            maxPixels=1e13,
            bestEffort=True,
        )
        .get("valid")
    )

    return ee.Image(img.set("ZZ_USER_VALID_PIXELS", valid_pixels))


def ee_cloud_probability_mask(img: ee.Image, threshold: float, invert: bool = False) -> ee.Image:
    mask = img.select(["cloud"]).gte(threshold) if invert else img.select(["cloud"]).lt(threshold)

    return img.updateMask(mask).select(img.bandNames().remove("cloud"))


def ee_gdf_to_feature_collection(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    gdf = gdf[["00_indexnum", "geometry", "start_date", "end_date"]]

    gdf["start_date"] = gdf["start_date"].dt.strftime("%Y-%m-%d")
    gdf["end_date"] = gdf["end_date"].dt.strftime("%Y-%m-%d")

    gdf.rename(
        columns={"start_date": "s", "end_date": "e", "00_indexnum": "0"}, inplace=True
    )  # saving memory when uploading geojson to GEE

    geo_json = os.path.join(os.getcwd(), "".join(random.choice(string.ascii_lowercase) for i in range(6)) + ".geojson")  # noqa: S311
    gdf = gdf.to_crs(4326)
    gdf.to_file(geo_json, driver="GeoJSON")

    with open(os.path.abspath(geo_json), encoding="utf-8") as f:
        json_dict = json.load(f)

    if json_dict["type"] == "FeatureCollection":
        for feature in json_dict["features"]:
            if feature["geometry"]["type"] != "Point":
                feature["geometry"]["geodesic"] = True
        features = ee.FeatureCollection(json_dict)

    os.remove(geo_json)

    return features


def ee_img_to_numpy(ee_img: ee.Image, ee_geometry: ee.Geometry, scale: int) -> np.ndarray:
    ee_img = ee.Image(ee_img)
    ee_geometry = ee.Geometry(ee_geometry).bounds()

    projection = ee.Projection("EPSG:4326").atScale(scale).getInfo()
    chip_size = round(ee_geometry.perimeter(0.1).getInfo() / (4 * scale))  # type: ignore  # noqa: PGH003

    scale_y = -projection["transform"][0]  # type: ignore  # noqa: PGH003
    scale_x = projection["transform"][4]  # type: ignore  # noqa: PGH003

    list_of_coordinates = ee.Array.cat(ee_geometry.coordinates(), 1).getInfo()

    x_min = list_of_coordinates[0][0]  # type: ignore  # noqa: PGH003
    y_max = list_of_coordinates[2][1]  # type: ignore  # noqa: PGH003
    coordinates = [x_min, y_max]

    chip_size = 1 if chip_size == 0 else chip_size

    img_in_bytes = ee.data.computePixels({
        "expression": ee_img,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": chip_size, "height": chip_size},
            "affineTransform": {
                "scaleX": scale_x,
                "scaleY": scale_y,
                "translateX": coordinates[0],
                "translateY": coordinates[1],
            },
            "crsCode": projection["crs"],  # type: ignore  # noqa: PGH003
        },
    })

    img_in_array = np.array(img_in_bytes.tolist()).astype(np.float32)
    img_in_array[np.isinf(img_in_array)] = 0
    img_in_array[np.isnan(img_in_array)] = 0

    return img_in_array


def ee_get_tasks_status() -> pd.DataFrame:
    tasks = ee.data.listOperations()

    if tasks:
        records = []
        for op in tasks:
            metadata = op.get("metadata", {})

            record = {
                "attempt": metadata.get("attempt"),
                "create_time": metadata.get("createTime"),
                "description": metadata.get("description"),
                "destination_uris": metadata.get("destinationUris", [None])[0],
                "done": op.get("done"),
                "end_time": metadata.get("endTime"),
                "name": op.get("name"),
                "priority": metadata.get("priority"),
                "progress": metadata.get("progress"),
                "script_uri": metadata.get("scriptUri"),
                "start_time": metadata.get("startTime"),
                "state": metadata.get("state"),
                "total_batch_eecu_usage_seconds": metadata.get("batchEecuUsageSeconds", 0.0),
                "type": metadata.get("type"),
                "update_time": metadata.get("updateTime"),
            }
            records.append(record)

        df = pd.DataFrame(records)
        df["create_time"] = pd.to_datetime(df.create_time, format="mixed")
        df["end_time"] = pd.to_datetime(df.end_time, format="mixed")
        df["start_time"] = pd.to_datetime(df.start_time, format="mixed")
        df["update_time"] = pd.to_datetime(df.update_time, format="mixed")

        df["estimated_cost_usd_tier_1"] = (df.total_batch_eecu_usage_seconds / (60 * 60)) * 0.40
        df["estimated_cost_usd_tier_2"] = (df.total_batch_eecu_usage_seconds / (60 * 60)) * 0.28
        df["estimated_cost_usd_tier_3"] = (df.total_batch_eecu_usage_seconds / (60 * 60)) * 0.16

    else:  # If no tasks are found, create an empty DataFrame with the same columns
        df = pd.DataFrame(
            columns=[
                "attempt",
                "create_time",
                "description",
                "destination_uris",
                "done",
                "end_time",
                "name",
                "priority",
                "progress",
                "script_uri",
                "start_time",
                "state",
                "total_batch_eecu_usage_seconds",
                "type",
                "update_time",
            ]
        )

    return df


def ee_get_reducers(reducer_names: list[str] | None = None) -> ee.Reducer:  # noqa: C901
    if reducer_names is None:
        reducer_names = ["median"]

    names = [n.lower() for n in reducer_names]

    pct_vals = sorted({int(n[1:]) for n in names if n.startswith("p")})

    reducers = []
    for n in names:
        if n == "min":
            reducers.append(ee.Reducer.min())
        elif n == "max":
            reducers.append(ee.Reducer.max())
        elif n == "mean":
            reducers.append(ee.Reducer.mean())
        elif n == "median":
            reducers.append(ee.Reducer.median())
        elif n == "kurt":
            reducers.append(ee.Reducer.kurtosis())
        elif n == "skew":
            reducers.append(ee.Reducer.skew())
        elif n == "std":
            reducers.append(ee.Reducer.stdDev())
        elif n == "var":
            reducers.append(ee.Reducer.variance())
        elif n == "mode":
            reducers.append(ee.Reducer.mode())
        elif n.startswith("p"):
            continue
        else:
            raise ValueError(f"Unknown reducer: '{n}'")  # noqa: TRY003

    if pct_vals:
        reducers.append(ee.Reducer.percentile(pct_vals))

    reducer = reducers[0]
    for r in reducers[1:]:
        reducer = reducer.combine(r, None, True)

    return reducer


def ee_filter_img_collection_invalid_pixels(
    ee_img_collection: ee.ImageCollection, ee_geometry: ee.Geometry, pixel_size: int, min_valid_pixels: int = 20
) -> ee.ImageCollection:
    min_valid_pixels = ee.Algorithms.If(
        ee_geometry.area(0.001),
        ee.Number(min_valid_pixels),
        ee.Number(1),
    )

    ee_img_collection = ee_img_collection.map(lambda i: ee_map_valid_pixels(i, ee_geometry, pixel_size)).filter(
        ee.Filter.gte("ZZ_USER_VALID_PIXELS", min_valid_pixels)
    )

    ee_img_collection = (
        ee_img_collection.map(lambda img: img.set("ZZ_USER_TIME_DUMMY", img.date().format("YYYY-MM-dd")))
        .sort("ZZ_USER_TIME_DUMMY")
        .distinct("ZZ_USER_TIME_DUMMY")
    )

    return ee_img_collection


def ee_get_number_of_pixels(ee_geometry: ee.Geometry, subsampling_max_pixels: float, pixel_size: int) -> ee.Number:
    # -- maxPixels logic (absolute or fraction of footprint) -- #
    if subsampling_max_pixels > 1:
        return ee.Number(subsampling_max_pixels)
    else:
        pixel_area = ee.Number(pixel_size).pow(2)
        total_pixels = ee_geometry.area(0.001).divide(pixel_area)
        return total_pixels.multiply(subsampling_max_pixels).toInt()


def ee_safe_remove_borders(ee_geometry: ee.Geometry, border_size: int, area_lower_bound: int) -> ee.Geometry:
    return ee.Geometry(
        ee.Algorithms.If(
            ee_geometry.buffer(-border_size, 0.001).area(0.001).gte(area_lower_bound),
            ee_geometry.buffer(-border_size, 0.001),
            ee_geometry,
        )
    )


def ee_add_indexes_to_image(image: ee.Image, indexes: list[str]) -> ee.Image:
    for index in indexes:
        calculated = image.expression(index, {"i": image})
        image = image.addBands(calculated, None, True)

    return image


def ee_quick_start() -> None:
    """Quick start function to initialize Earth Engine."""

    if "GEE_KEY" in os.environ:
        gee_key = os.environ["GEE_KEY"]

        if gee_key.endswith(".json"):  # Corporative usage
            credentials = ee.ServiceAccountCredentials(gee_key, gee_key)
            ee.Initialize(credentials)

            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gee_key

            with open(gee_key) as f:
                key_data = json.load(f)
                print(
                    f"Earth Engine initialized successfully using AgriGEE.lite for corporative usage. Project: {key_data.get('project_id', 'Unknown')}, Email: {key_data.get('client_email', 'Unknown')}."
                )

        else:  # Academic usage
            ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project=gee_key)
            print(f"Earth Engine initialized successfully using AgriGEE.lite for academic usage (project={gee_key}).")

    else:
        print(
            "Earth Engine not initialized. Please set the GEE_KEY environment variable to your Earth Engine key. You can find more information in the AgriGEE.lite documentation."
        )
