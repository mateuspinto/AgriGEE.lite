import hashlib
import inspect
import json
from pathlib import Path
from typing import cast

import geopandas as gpd
import h3
import numpy as np
import pandas as pd
import polars as pl
from shapely.geometry import MultiPolygon, Point, Polygon
from tqdm.auto import tqdm

from agrigee_lite._geo_compat import (
    GeoDataFrameLike,
    get_crs,
    iter_shapely_geometries,
    normalize_geodataframe,
    restore_geodataframe_type,
    wrap_geopolars_frame,
)


def simplify_gdf(gdf: GeoDataFrameLike, tol: float = 0.001, crs: str | None = None) -> GeoDataFrameLike:
    """
    Simplify geometries in a compatible geo frame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopolars.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    tol : float, optional
        Tolerance for simplification (default is 0.001).

    Returns
    -------
    geopandas.GeoDataFrame or geopolars.GeoDataFrame
        GeoDataFrame with unchanged geometries for now.

    Notes
    -----
    TODO: reintroduce geometry simplification here once GeoPolars exposes a
    stable native simplify API.
    """
    _ = tol
    normalized = normalize_geodataframe(gdf, crs=crs)
    return restore_geodataframe_type(gdf, normalized, preserve_index=True)


def h3_clustering(
    gdf: GeoDataFrameLike,
    coarse_resolution: int = 5,
    fine_resolution: int = 8,
    crs: str | None = None,
) -> GeoDataFrameLike:
    """
    Cluster and order geometries using H3 hierarchical spatial indexing.

    Each geometry's centroid is mapped to two H3 resolutions:

    - **coarse**: defines cluster boundaries — geometries in the same coarse cell
      share a ``cluster_id``.  A chunk that would span multiple coarse cells is
      split into separate clusters, so every GEE request covers one contiguous
      geographic region.
    - **fine**: used only for ordering within a cluster, ensuring consecutive
      rows are as spatially close as possible.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame or geopolars.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    coarse_resolution : int, optional
        H3 resolution for cluster boundaries (default 5, ≈14 km edge length).
    fine_resolution : int, optional
        H3 resolution for intra-cluster ordering (default 8, ≈460 m edge length).

    Returns
    -------
    geopandas.GeoDataFrame or geopolars.GeoDataFrame
        GeoDataFrame with ``cluster_id`` column (integer, one per coarse cell),
        sorted by ``(coarse_cell, fine_cell)``, without geometry simplification
        for now.
    """
    normalized = normalize_geodataframe(gdf, crs=crs)
    geometries = iter_shapely_geometries(normalized)

    fine_cells: list[str] = []
    coarse_cells: list[str] = []

    for geometry in tqdm(geometries, desc="H3 cells", leave=True):
        centroid = cast(Point, geometry if geometry.geom_type == "Point" else geometry.centroid)
        fine_cell = h3.latlng_to_cell(centroid.y, centroid.x, fine_resolution)
        fine_cells.append(fine_cell)
        coarse_cells.append(h3.cell_to_parent(fine_cell, coarse_resolution))

    frame = normalized.with_row_index("_row_idx")
    frame = frame.with_columns(
        pl.Series("_h3_fine", fine_cells),
        pl.Series("_h3_coarse", coarse_cells),
    )
    frame = frame.sort(["_h3_coarse", "_h3_fine", "_row_idx"])

    cluster_ids: list[int] = []
    cluster_by_cell: dict[str, int] = {}
    for coarse_cell in frame.get_column("_h3_coarse").to_list():
        cluster_ids.append(cluster_by_cell.setdefault(str(coarse_cell), len(cluster_by_cell)))

    frame = frame.with_columns(pl.Series("cluster_id", cluster_ids))
    frame = frame.rename({"_h3_coarse": "h3_coarse", "_h3_fine": "h3_fine"}).drop("_row_idx")

    # TODO: reintroduce geometry simplification here once GeoPolars exposes a
    # stable native simplify API.
    clustered = wrap_geopolars_frame(frame, crs=get_crs(normalized))
    return restore_geodataframe_type(gdf, clustered)


def create_gdf_hash(
    gdf: GeoDataFrameLike,
    start_date_column_name: str,
    end_date_column_name: str,
    crs: str | None = None,
) -> str:
    normalized = normalize_geodataframe(gdf, crs=crs)
    geometries = iter_shapely_geometries(normalized)
    date_rows = normalized.select([start_date_column_name, end_date_column_name]).iter_rows(named=True)

    hash_rows = []
    for geometry, row in zip(geometries, date_rows):
        centroid = geometry.centroid
        hash_rows.append({
            start_date_column_name: row[start_date_column_name],
            end_date_column_name: row[end_date_column_name],
            "centroid_x": centroid.x,
            "centroid_y": centroid.y,
        })

    hash_values = pd.util.hash_pandas_object(pd.DataFrame(hash_rows)).values
    return hashlib.sha1(hash_values).hexdigest()  # type: ignore  # noqa: PGH003, S324


def create_dict_hash(d: dict) -> str:
    """
    Create a hash for a dictionary, normalizing sets to sorted lists.

    Parameters
    ----------
    d : dict
        Dictionary to hash.

    Returns
    -------
    str
        SHA1 hash string representing the contents of the dictionary.
    """

    def convert_sets_to_sorted_lists(obj):
        if isinstance(obj, dict):
            return {k: convert_sets_to_sorted_lists(v) for k, v in obj.items()}
        elif isinstance(obj, set):
            return sorted(obj)
        elif isinstance(obj, list):
            return [convert_sets_to_sorted_lists(i) for i in obj]
        else:
            return obj

    normalized = convert_sets_to_sorted_lists(d)
    return hashlib.sha1(json.dumps(normalized, sort_keys=True).encode("utf-8")).hexdigest()  # noqa: S324


def log_dict_function_call_summary(ignore: list[str] | None = None) -> dict[str, dict[str, str]]:
    """
    Log a summary of function call arguments as a dictionary.

    Parameters
    ----------
    ignore : list of str, optional
        List of argument names to ignore (default is None).

    Returns
    -------
    dict
        Dictionary mapping function name to argument values.
    """
    _current = inspect.currentframe()
    assert _current is not None
    frame = _current.f_back
    assert frame is not None
    func_name = frame.f_code.co_name
    args, _, _, values = inspect.getargvalues(frame)
    ignore = ignore or []
    args_dict = {str(arg): str(values[arg]) for arg in args if arg not in ignore}
    return {func_name: args_dict}


def create_grid_centroids_numpy(geometry: Polygon | MultiPolygon | Point, n_cells: int = 10) -> np.ndarray:
    """
    Generate grid centroids within a geometry.

    Parameters
    ----------
    geometry : shapely.geometry.Polygon or MultiPolygon or Point
        Geometry to generate centroids for.
    n_cells : int, optional
        Number of centroids to generate (default is 10).

    Returns
    -------
    numpy.ndarray
        Array of centroid coordinates (shape: [n_cells, 2]).
    """
    try:
        xmin, ymin, xmax, ymax = geometry.bounds
        cell_size = (xmax - xmin) / n_cells

        num_cols = int(np.ceil((xmax - xmin) / cell_size))
        num_rows = int(np.ceil((ymax - ymin) / cell_size))
        max_points = num_cols * num_rows

        centroids = np.empty((max_points, 2), dtype=np.float32)
        count = 0

        for x in np.arange(xmin + cell_size / 2, xmax, cell_size):
            for y in np.arange(ymin + cell_size / 2, ymax, cell_size):
                point = Point(x, y)
                if geometry.contains(point):
                    centroids[count] = [x, y]
                    count += 1

        if count >= n_cells:
            return centroids[np.random.choice(count, size=n_cells, replace=False)]
        else:  # count < n_cells
            return np.zeros((n_cells, 2), dtype=np.float32)
    except:  # noqa: E722
        return np.zeros((n_cells, 2), dtype=np.float32)


def generate_grid_random_points_from_gdf(gdf: gpd.GeoDataFrame, num_points_per_geometry: int = 10) -> gpd.GeoDataFrame:
    """
    Generate random grid points for each geometry in a GeoDataFrame.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    num_points_per_geometry : int, optional
        Number of points to generate per geometry (default is 10).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of generated points with geometry_id.
    """
    centroids = np.empty((num_points_per_geometry * gdf.geometry.nunique(), 2), dtype=np.float32)
    geometry_ids = np.empty((num_points_per_geometry * gdf.geometry.nunique()), dtype=np.int32)

    for n, (_, row) in enumerate(
        tqdm(gdf[["geometry", "geometry_id"]].drop_duplicates().iterrows(), total=gdf.geometry.nunique())
    ):
        geom = row.geometry
        geometry_id = row.geometry_id
        centroids_sub = create_grid_centroids_numpy(geom, n_cells=num_points_per_geometry)
        centroids[n * num_points_per_geometry : (n + 1) * num_points_per_geometry, :] = centroids_sub
        geometry_ids[n * num_points_per_geometry : (n + 1) * num_points_per_geometry] = geometry_id

    gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat) for lon, lat in centroids], crs=gdf.crs)
    gdf["geometry_id"] = geometry_ids

    return gdf


def random_points_from_gdf(
    gdf: gpd.GeoDataFrame, num_points_per_geometry: int = 10, buffer: int = -10
) -> gpd.GeoDataFrame:
    """
    Generate random points from geometries in a GeoDataFrame, with optional buffering.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    num_points_per_geometry : int, optional
        Number of points to generate per geometry (default is 10).
    buffer : int, optional
        Buffer distance to apply to geometries before generating points (default is -10).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame of generated points merged with original attributes.
    """
    if buffer != 0:
        gdf = gdf.copy()
        gdf = cast(gpd.GeoDataFrame, h3_clustering(gdf, crs=str(gdf.crs) if gdf.crs is not None else None))
        gdf["geometry"] = gdf.to_crs(gdf.estimate_utm_crs()).buffer(buffer).to_crs("EPSG:4326")

    gdf["geometry_id"] = pd.factorize(gdf["geometry"])[0]
    points_gdf = generate_grid_random_points_from_gdf(gdf, num_points_per_geometry)
    points_gdf = points_gdf.merge(
        gdf.drop(columns=["geometry"]).reset_index().rename(columns={"index": "original_index"}),
        on="geometry_id",
        how="inner",
    )
    points_gdf = points_gdf[points_gdf.geometry.x != 0].reset_index(drop=True)

    return points_gdf


def get_sample_gdf() -> gpd.GeoDataFrame:
    """
    Load the bundled sample GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Sample GeoDataFrame for testing and demonstration purposes.
    """
    return gpd.read_parquet(Path(__file__).parent / "data" / "sample.parquet")


def get_reducer_names(reducer_names: set[str] | None = None) -> list[str]:
    """
    Get standardized reducer names from a set of reducer names.

    Parameters
    ----------
    reducer_names : set of str, optional
        Set of reducer names (default is {"median"}).

    Returns
    -------
    list of str
        List of standardized reducer names.
    """
    if reducer_names is None:
        reducer_names = {"median"}

    names = sorted([n.lower() for n in reducer_names])

    pct_vals = sorted({int(n[1:]) for n in names if n.startswith("p")})

    reducers = []
    for n in names:
        if n in {"min", "max", "mean", "median", "mode"}:
            reducers.append(n)
        elif n == "kurt":
            reducers.append("kurtosis")
        elif n == "skew":
            reducers.append("skew")
        elif n == "std":
            reducers.append("stdDev")
        elif n == "var":
            reducers.append("variance")
        elif n.startswith("p"):
            continue
        else:
            raise ValueError(f"Unknown reducer: '{n}'")  # noqa: TRY003

    for v in pct_vals:
        reducers.append(f"p{v}")

    return reducers
