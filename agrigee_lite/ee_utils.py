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
    """
    Extract band statistics from an Earth Engine image for a given feature geometry.

    This function applies statistical reducers to all bands of an Earth Engine image
    within the geometry of a given feature. It handles metadata extraction including
    timestamps and valid pixel counts, making it suitable for time series analysis.

    Parameters
    ----------
    ee_img : ee.Image
        Earth Engine image to process.
    ee_feature : ee.Feature
        Earth Engine feature providing the geometry for spatial reduction.
    pixel_size : int
        Spatial resolution in meters for the reduction operation.
    subsampling_max_pixels : ee.Number
        Maximum number of pixels to include in the reduction.
    reducer : ee.Reducer
        Earth Engine reducer to apply (e.g., median, mean, max).
    single_image : bool, optional
        Whether to skip timestamp extraction for single image processing, by default False.

    Returns
    -------
    ee.Feature
        Earth Engine feature containing the reduced statistics as properties.
        Includes band values, timestamp (if not single_image), index number, and valid pixel count.
    """
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
    """
    Add valid pixel count metadata to an Earth Engine image.

    Counts the number of valid (non-cloud/non-satelliteError) pixels within a geometry and adds this
    information as metadata to the image. This is useful for quality assessment
    and filtering images with insufficient valid data.

    Parameters
    ----------
    img : ee.Image
        Input Earth Engine image to analyze.
    ee_geometry : ee.Geometry
        Geometry defining the area of interest for pixel counting.
    pixel_size : int
        Spatial resolution in meters for the pixel counting operation.

    Returns
    -------
    ee.Image
        The original image with added "ZZ_USER_VALID_PIXELS" property containing
        the count of valid pixels within the specified geometry.
    """
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
    """
    Apply cloud probability masking to an Earth Engine image.

    Masks pixels based on cloud probability values, typically from cloud detection
    algorithms. Pixels with cloud probability above (or below if inverted) the
    threshold are masked out. The cloud band is removed from the output.

    Parameters
    ----------
    img : ee.Image
        Input image containing a "cloud" band with probability values.
    threshold : float
        Cloud probability threshold (0.0 to 1.0 or 0 to 100 depending on data).
    invert : bool, optional
        If True, mask pixels with probability >= threshold (typical use).
        If False, mask pixels with probability < threshold, by default False.

    Returns
    -------
    ee.Image
        Masked image with cloud band removed and cloud pixels masked out.
    """
    mask = img.select(["cloud"]).gte(threshold) if invert else img.select(["cloud"]).lt(threshold)

    return img.updateMask(mask).select(img.bandNames().remove("cloud"))


def ee_gdf_to_feature_collection(
    gdf: gpd.GeoDataFrame,
    original_index_column_name: str,
    start_date_column_name: str = "start_date",
    end_date_column_name: str = "end_date",
) -> ee.FeatureCollection:
    """
    Convert a GeoPandas GeoDataFrame to an Earth Engine FeatureCollection.

    Transforms a GeoDataFrame with temporal and spatial information into an Earth Engine
    FeatureCollection suitable for server-side processing. The function handles coordinate
    system conversion, temporary file creation, and proper geodesic geometry settings.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input GeoDataFrame containing geometries and temporal information.
    original_index_column_name : str
        Name of the column containing original feature indices.
    start_date_column_name : str, optional
        Name of the start date column, by default "start_date".
    end_date_column_name : str, optional
        Name of the end date column, by default "end_date".

    Returns
    -------
    ee.FeatureCollection
        Earth Engine FeatureCollection with renamed properties:
        - Original index column → "0"
        - Start date column → "s"
        - End date column → "e"
        Non-point geometries are automatically set as geodesic for accurate processing.

    Notes
    -----
    This function creates temporary GeoJSON files that are automatically cleaned up.
    Input CRS is converted to EPSG:4326 for Earth Engine compatibility.
    """
    gdf = gdf.copy()

    gdf = gdf[[original_index_column_name, "geometry", start_date_column_name, end_date_column_name]]

    gdf[start_date_column_name] = gdf[start_date_column_name].dt.strftime("%Y-%m-%d")
    gdf[end_date_column_name] = gdf[end_date_column_name].dt.strftime("%Y-%m-%d")

    gdf.rename(
        columns={start_date_column_name: "s", end_date_column_name: "e", original_index_column_name: "0"}, inplace=True
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
    """
    Convert an Earth Engine image to a NumPy array for local processing.

    Downloads an Earth Engine image within a specified geometry and converts it
    to a NumPy array. The function handles coordinate system transformations,
    determines appropriate chip size, and cleans invalid values (NaN, Inf).

    Parameters
    ----------
    ee_img : ee.Image
        Earth Engine image to convert.
    ee_geometry : ee.Geometry
        Geometry defining the spatial extent for image extraction.
        Will be converted to bounds for rectangular extraction.
    scale : int
        Spatial resolution in meters for the downloaded image.

    Returns
    -------
    np.ndarray
        NumPy array containing the image data as float32.
        Invalid values (NaN, Inf) are replaced with 0.
        Shape depends on the geometry size and requested scale.

    Notes
    -----
    The function automatically calculates appropriate chip dimensions based on
    geometry perimeter and scale. Very small geometries will result in 1x1 chips.
    """
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
    """
    Retrieve status information for all Earth Engine tasks.

    Fetches comprehensive information about all Earth Engine operations/tasks
    associated with the authenticated account, including metadata, timing,
    resource usage, and cost estimates.

    Returns
    -------
    pd.DataFrame
        DataFrame containing task information with the following columns:
        - attempt: Task attempt number
        - create_time: Task creation timestamp
        - description: Task description/name
        - destination_uris: Output destination URIs
        - done: Boolean indicating completion status
        - end_time: Task completion timestamp
        - name: Internal task name
        - priority: Task priority level
        - progress: Completion progress (0.0 to 1.0)
        - script_uri: Source script URI
        - start_time: Task start timestamp
        - state: Current task state (RUNNING, COMPLETED, FAILED, etc.)
        - total_batch_eecu_usage_seconds: Total EECU usage in seconds
        - type: Task type (EXPORT_IMAGE, EXPORT_TABLE, etc.)
        - update_time: Last update timestamp
        - estimated_cost_usd_tier_1: Estimated cost in US Dollars for Tier 1 pricing
        - estimated_cost_usd_tier_2: Estimated cost in US Dollars for Tier 2 pricing
        - estimated_cost_usd_tier_3: Estimated cost in US Dollars for Tier 3 pricing

    Notes
    -----
    Cost estimates are based on EECU usage and standard pricing tiers.
    If no tasks exist, returns an empty DataFrame with the same column structure.
    """
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


def ee_get_reducers(reducer_names: set[str] | None = None) -> ee.Reducer:  # noqa: C901
    """
    Create a combined Earth Engine reducer from a set of reducer names.

    Builds a composite Earth Engine reducer by combining multiple statistical
    reducers. Supports standard statistics and percentiles, with automatic
    handling of percentile combinations.

    Parameters
    ----------
    reducer_names : set[str] or None, optional
        Set of reducer names to combine. Supported values:
        - "min", "max", "mean", "median"
        - "kurt" (kurtosis), "skew" (skewness)
        - "std" (standard deviation), "var" (variance)
        - "mode" (most frequent value)
        - "pXX" (percentiles, e.g., "p10", "p90")
        If None, defaults to ["median"], by default None.

    Returns
    -------
    ee.Reducer
        Combined Earth Engine reducer that applies all requested statistics.
        Multiple percentiles are automatically grouped into a single percentile reducer.

    Raises
    ------
    ValueError
        If an unsupported reducer name is provided.

    Examples
    --------
    >>> # Single reducer
    >>> reducer = ee_get_reducers({"median"})
    >>> # Multiple reducers including percentiles
    >>> reducer = ee_get_reducers({"mean", "std", "p10", "p90"})
    """
    if reducer_names is None:
        reducer_names = ["median"]

    names = sorted([n.lower() for n in reducer_names])

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
    """
    Filter an Earth Engine ImageCollection based on valid pixel count and temporal uniqueness.

    Removes images with insufficient valid pixels and ensures temporal uniqueness
    by keeping only the first image per date. This is essential for time series
    analysis where data quality and temporal consistency are important.

    Parameters
    ----------
    ee_img_collection : ee.ImageCollection
        Input Earth Engine ImageCollection to filter.
    ee_geometry : ee.Geometry
        Geometry defining the area of interest for pixel counting.
    pixel_size : int
        Spatial resolution in meters for pixel counting operations.
    min_valid_pixels : int, optional
        Minimum number of valid pixels required per image, by default 20.
        For very small geometries, this is automatically adjusted to 1.

    Returns
    -------
    ee.ImageCollection
        Filtered ImageCollection with:
        - Images having at least min_valid_pixels valid pixels
        - Temporal uniqueness (one image per date)
        - Added "ZZ_USER_TIME_DUMMY" property with YYYY-MM-DD format dates
        - Sorted by date in ascending order

    Notes
    -----
    The function automatically adjusts the minimum pixel threshold for small areas.
    Each image gets a "ZZ_USER_VALID_PIXELS" property added during processing.
    """
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
    """
    Calculate the maximum number of pixels for Earth Engine reduction operations.

    Determines the appropriate maxPixels value for Earth Engine reducers based on
    either absolute pixel count or fractional area sampling. This helps control
    memory usage and processing time for large geometries.

    Parameters
    ----------
    ee_geometry : ee.Geometry
        Geometry defining the area of interest.
    subsampling_max_pixels : float
        Maximum pixels specification:
        - If > 1: Used as absolute pixel count
        - If ≤ 1: Used as fraction of total geometry pixels (0.1 = 10% sampling)
    pixel_size : int
        Spatial resolution in meters for pixel area calculation.

    Returns
    -------
    ee.Number
        Earth Engine Number representing the maximum pixels to use in reductions.
        Either the absolute value (if > 1) or calculated fraction of total pixels.

    Examples
    --------
    >>> # Absolute pixel count
    >>> max_pixels = ee_get_number_of_pixels(geometry, 10000, 30)
    >>> # 50% sampling of total area
    >>> max_pixels = ee_get_number_of_pixels(geometry, 0.5, 30)
    """
    # -- maxPixels logic (absolute or fraction of footprint) -- #
    if subsampling_max_pixels > 1:
        return ee.Number(subsampling_max_pixels)
    else:
        pixel_area = ee.Number(pixel_size).pow(2)
        total_pixels = ee_geometry.area(0.001).divide(pixel_area)
        return total_pixels.multiply(subsampling_max_pixels).toInt()


def ee_safe_remove_borders(ee_geometry: ee.Geometry, border_size: int, area_lower_bound: int) -> ee.Geometry:
    """
    Safely apply negative buffer to remove geometry borders with area validation.

    Applies a negative buffer to remove border pixels from a geometry, but only
    if the resulting geometry maintains sufficient area. This prevents creating
    overly small or invalid geometries while reducing edge effects.

    Parameters
    ----------
    ee_geometry : ee.Geometry
        Input geometry to process.
    border_size : int
        Size of border to remove in meters (applied as negative buffer).
    area_lower_bound : int
        Minimum area threshold in square meters below which borders won't be removed.

    Returns
    -------
    ee.Geometry
        Processed geometry:
        - If buffered area ≥ area_lower_bound: geometry with borders removed
        - If buffered area < area_lower_bound: original geometry unchanged

    Notes
    -----
    This function helps avoid issues with satellite data processing where border
    pixels might have quality issues, while ensuring geometries remain valid
    and sufficiently large for analysis.
    """
    return ee.Geometry(
        ee.Algorithms.If(
            ee_geometry.buffer(-border_size, 0.001).area(0.001).gte(area_lower_bound),
            ee_geometry.buffer(-border_size, 0.001),
            ee_geometry,
        )
    )


def ee_add_indexes_to_image(image: ee.Image, indexes: list[str]) -> ee.Image:
    """
    Add vegetation indices as bands to an Earth Engine image.

    Calculates specified vegetation indices using image expressions and adds them
    as new bands to the input image. Each index is computed using the image bands
    available at the time of calculation.

    Parameters
    ----------
    image : ee.Image
        Input Earth Engine image containing the necessary bands for index calculation.
    indexes : list[str]
        List of index expressions to calculate and add as bands.
        Each expression should be a valid Earth Engine expression using 'i'
        to reference the input image.

    Returns
    -------
    ee.Image
        Enhanced image with original bands plus calculated vegetation indices.
        New index bands are added with names matching the expression strings.

    Examples
    --------
    >>> # Add NDVI and EVI indices
    >>> indices = ["(i.nir - i.red) / (i.nir + i.red)", "2.5 * ((i.nir - i.red) / (i.nir + 6*i.red - 7.5*i.blue + 1))"]
    >>> enhanced_image = ee_add_indexes_to_image(image, indices)

    Notes
    -----
    The function uses Earth Engine's expression() method where 'i' represents
    the input image. Band names in expressions should match the actual band
    names in the image (e.g., 'i.red', 'i.nir').
    """
    for index in indexes:
        calculated = image.expression(index, {"i": image})
        image = image.addBands(calculated, None, True)

    return image


def ee_is_authenticated() -> bool:
    """
    Check if Earth Engine is properly authenticated and initialized.

    Attempts to initialize Earth Engine and returns whether the operation
    was successful. This is useful for checking authentication status
    before performing Earth Engine operations.

    Returns
    -------
    bool
        True if Earth Engine is successfully initialized and authenticated,
        False if initialization fails due to authentication or other issues.

    Examples
    --------
    >>> if ee_is_authenticated():
    >>>     print("Earth Engine is ready!")
    >>> else:
    >>>     print("Authentication required")
    """
    try:
        ee.Initialize()
    except Exception:
        return False
    else:
        return True


def ee_quick_start() -> None:
    """
    Quick start function to initialize Earth Engine with automatic credential detection.

    Automatically detects and uses Earth Engine credentials from the GEE_KEY
    environment variable. Supports both service account JSON files and project
    tokens, providing informative feedback about the initialization process.

    Environment Variables
    ---------------------
    GEE_KEY : str
        Earth Engine authentication key. Can be either:
        - Path to service account JSON file (ends with .json)
        - Project token string for standard authentication

    Returns
    -------
    None
        Prints initialization status messages but doesn't return values.

    Examples
    --------
    >>> import os
    >>> os.environ['GEE_KEY'] = '/path/to/service-account.json'
    >>> ee_quick_start()
    Earth Engine initialized successfully using AgriGEE.lite...

    Notes
    -----
    For service account authentication, the function also sets the
    GOOGLE_APPLICATION_CREDENTIALS environment variable for use Google Cloud Storage.
    """

    if not ee_is_authenticated():
        if "GEE_KEY" in os.environ:
            gee_key = os.environ["GEE_KEY"]

            if gee_key.endswith(".json"):  # with service account
                credentials = ee.ServiceAccountCredentials(gee_key, gee_key)
                ee.Initialize(credentials)

                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gee_key

                with open(gee_key) as f:
                    key_data = json.load(f)
                    print(
                        f"Earth Engine initialized successfully using AgriGEE.lite with service account. Project: {key_data.get('project_id', 'Unknown')}, Email: {key_data.get('client_email', 'Unknown')}."
                    )

            else:  # using token
                ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project=gee_key)
                print(f"Earth Engine initialized successfully using AgriGEE.lite using token (project={gee_key}).")

        else:
            print(
                "Earth Engine not initialized. Please set the GEE_KEY environment variable to your Earth Engine key. You can find more information in the AgriGEE.lite documentation."
            )
