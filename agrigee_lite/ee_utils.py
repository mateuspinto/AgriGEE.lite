import json
import os
import random
import string

import ee
import geopandas as gpd


def ee_map_bands_and_doy(
    ee_img: ee.Image,
    ee_geometry: ee.Geometry,
    ee_feature: ee.Feature,
    scale: int,
    round_int_16: bool = False,
    max_pixels: int = 1000,
) -> ee.Feature:
    ee_img = ee.Image(ee_img)
    stats = ee_img.reduceRegion(
        reducer=ee.Reducer.median(),
        geometry=ee_geometry,
        scale=scale,
        maxPixels=max_pixels,
        bestEffort=True,
    ).map(lambda _, value: ee.Number(ee.Algorithms.If(ee.Algorithms.IsEqual(value, None), 0, value)))

    if round_int_16:
        stats = stats.map(lambda _, value: ee.Number(value).round())

    stats = stats.set("doy", ee_img.date().getRelative("day", "year").add(1)).set(
        "index_num", ee_feature.get("index_num")
    )

    return ee.Feature(None, stats)


def ee_map_valid_pixels(img: ee.Image, ee_geometry: ee.Geometry, scale: float) -> ee.Image:
    mask = ee.Image(img).select([0]).gt(0)

    valid_pixels = ee.Number(
        mask.rename("valid")
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=ee_geometry,
            scale=scale,
            maxPixels=1e8,
            bestEffort=True,
        )
        .get("valid")
    )

    return ee.Image(img.set("ZZ_USER_VALID_PIXELS", valid_pixels))


def ee_cloud_probability_mask(img: ee.Image, threshold: float, invert: bool = False) -> ee.Image:
    mask = img.select(["cloud"]).gte(threshold) if invert else img.select(["cloud"]).lt(threshold)

    return img.updateMask(mask).select(img.bandNames().remove("cloud"))


def ee_gdf_to_feature_collection(gdf: gpd.GeoDataFrame) -> ee.FeatureCollection:
    gdf = gdf.copy()
    gdf = gdf[["geometry", "start_date", "end_date"]]

    gdf["index_num"] = gdf.index.values.astype(int)
    gdf["start_date"] = gdf["start_date"].dt.strftime("%Y-%m-%d")
    gdf["end_date"] = gdf["end_date"].dt.strftime("%Y-%m-%d")

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
