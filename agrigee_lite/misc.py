import concurrent.futures
import hashlib
import inspect
import json
import warnings
from collections import deque

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from topojson import Topology
from tqdm.std import tqdm


def build_quadtree_iterative(gdf: gpd.GeoDataFrame, max_size: int = 1000) -> list[list[int]]:
    """
    Build a quadtree iteratively to cluster geometries into groups of max_size.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    max_size : int, optional
        Maximum number of geometries per cluster (default is 1000).

    Returns
    -------
    list of list of int
        List of clusters, each cluster is a list of indices of geometries in gdf.
    """
    queue: deque[tuple[gpd.GeoDataFrame, int]] = deque()
    queue.append((gdf, 0))
    leaves = []

    while queue:
        subset, depth = queue.popleft()
        n = len(subset)
        if n <= max_size:
            leaves.append(subset.index.to_numpy())
            continue

        dim = "centroid_x" if depth % 2 == 0 else "centroid_y"

        subset_sorted = subset.sort_values(by=dim)
        median_idx = n // 2
        median_val = subset_sorted.iloc[median_idx][dim]

        left = subset_sorted[subset_sorted[dim] <= median_val]
        right = subset_sorted[subset_sorted[dim] > median_val]

        queue.append((left, depth + 1))
        queue.append((right, depth + 1))

    return leaves


def build_quadtree(gdf: gpd.GeoDataFrame, max_size: int = 1000, depth: int = 0) -> list[list[int]]:
    """
    Build a quadtree recursively to cluster geometries into groups of max_size.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    max_size : int, optional
        Maximum number of geometries per cluster (default is 1000).
    depth : int, optional
        Current depth in the quadtree (default is 0).

    Returns
    -------
    list of list of int
        List of clusters, each cluster is a list of indices of geometries in gdf.
    """
    n = len(gdf)
    if n <= max_size:
        return [gdf.index.to_numpy()]

    dim = "centroid_x" if depth % 2 == 0 else "centroid_y"

    gdf_sorted = gdf.sort_values(by=dim)

    median_idx = n // 2
    median_val = gdf_sorted.iloc[median_idx][dim]

    left = gdf_sorted[gdf_sorted[dim] <= median_val]
    right = gdf_sorted[gdf_sorted[dim] > median_val]

    left_clusters = build_quadtree(left, max_size, depth + 1)
    right_clusters = build_quadtree(right, max_size, depth + 1)

    return left_clusters + right_clusters


def simplify_gdf(gdf: gpd.GeoDataFrame, tol: float = 0.001) -> gpd.GeoDataFrame:
    """
    Simplify geometries in a GeoDataFrame using TopoJSON, avoiding duplicates.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    tol : float, optional
        Tolerance for simplification (default is 0.001).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with simplified geometries.
    """
    """
    1. Detect duplicate geometries once, using WKB-hex as a stable key.
    2. Run TopoJSON simplification only on the unique geometries.
    3. Propagate the simplified result back to every original row.
    """
    gdf = gdf.copy()

    # ------------------------------------------------------------------
    # 1.  Build a geometry-only frame and keep just the unique geometries
    # ------------------------------------------------------------------
    gdf["_geom_key"] = gdf.geometry.apply(lambda g: g.wkb_hex)  # fast, deterministic
    unique_gdf = gdf[["_geom_key", "geometry"]].drop_duplicates("_geom_key")

    # ---------------------------------------------------------------
    # 2.  Simplify the unique geometries once with Topology.toposimplify
    # ---------------------------------------------------------------
    topo = Topology(unique_gdf[["geometry"]], prequantize=False)
    topo = topo.toposimplify(tol, prevent_oversimplify=True)
    simplified_unique = topo.to_gdf()

    # topo.to_gdf() returns rows in the same order, so align keys back
    simplified_unique["_geom_key"] = unique_gdf["_geom_key"].values

    # -------------------------------------------------------
    # 3.  Merge the simplified geometries back to the original
    # -------------------------------------------------------
    out = (
        gdf.drop(columns="geometry")
        .merge(simplified_unique[["_geom_key", "geometry"]], on="_geom_key", how="left")
        .drop(columns="_geom_key")
        .set_geometry("geometry")
    )
    out.index = gdf.index  # keep the original ordering
    return out


def _simplify_cluster(cluster: gpd.GeoDataFrame, cluster_id: int) -> tuple[int, gpd.GeoSeries]:
    """
    Simplify geometries in a cluster and return the cluster id and simplified geometries.

    Parameters
    ----------
    cluster : geopandas.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    cluster_id : int
        Identifier for the cluster.

    Returns
    -------
    tuple of (int, geopandas.GeoSeries)
        Cluster id and simplified geometries.
    """
    simplified = simplify_gdf(cluster)
    return cluster_id, simplified.geometry


def quadtree_clustering(gdf: gpd.GeoDataFrame, max_size: int = 1000) -> gpd.GeoDataFrame:
    """
    Cluster geometries in a GeoDataFrame using a quadtree and simplify clusters.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        GeoDataFrame containing geometries (Polygon, MultiPolygon, or Point).
    max_size : int, optional
        Maximum number of geometries per cluster (default is 1000).

    Returns
    -------
    geopandas.GeoDataFrame
        GeoDataFrame with cluster labels and simplified geometries.
    """
    gdf = gdf.copy()

    # Centroid columns (ignore CRS warning)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        gdf["centroid_x"] = gdf.geometry.centroid.x
        gdf["centroid_y"] = gdf.geometry.centroid.y

    # Build quadtree and label clusters
    clusters = build_quadtree_iterative(gdf, max_size=max_size)

    cluster_array = np.zeros(len(gdf), dtype=int)
    for i, cluster_indexes in enumerate(clusters):
        cluster_array[cluster_indexes] = i

    gdf["cluster_id"] = cluster_array
    gdf = gdf.sort_values(by=["cluster_id", "centroid_x"]).reset_index(drop=True)

    unique_cluster_ids = gdf["cluster_id"].unique()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(_simplify_cluster, gdf[gdf.cluster_id == cluster_id][["geometry"]], cluster_id): cluster_id
            for cluster_id in unique_cluster_ids
        }

        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Simplifying clusters",
            smoothing=0.5,
        ):
            cluster_id, simplified_geom = future.result()
            gdf.loc[gdf["cluster_id"] == cluster_id, "geometry"] = simplified_geom.values

    return gdf


def create_gdf_hash(gdf: gpd.GeoDataFrame, start_date_column_name: str, end_date_column_name: str) -> str:
    gdf_copy = gdf[["geometry", start_date_column_name, end_date_column_name]].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf_copy["centroid_x"] = gdf_copy.geometry.centroid.x
        gdf_copy["centroid_y"] = gdf_copy.geometry.centroid.y

    gdf_copy = gdf_copy.drop(columns=["geometry"])

    hash_values = pd.util.hash_pandas_object(gdf_copy).values
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
    frame = inspect.currentframe().f_back
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
        if count == n_cells:
            return centroids[np.random.choice(count, size=n_cells, replace=True)]
        else:  # count < n_cells:
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
        gdf = quadtree_clustering(gdf)
        gdf["geometry"] = gdf.to_crs(gdf.estimate_utm_crs()).buffer(-10).to_crs("EPSG:4326")

    gdf["geometry_id"] = pd.factorize(gdf["geometry"])[0]
    points_gdf = generate_grid_random_points_from_gdf(gdf, num_points_per_geometry)
    points_gdf = points_gdf.merge(
        gdf.drop(columns=["geometry"]).reset_index().rename(columns={"index": "original_index"}),
        on="geometry_id",
        how="inner",
    )
    points_gdf = points_gdf[points_gdf.geometry.x != 0].reset_index(drop=True)

    return points_gdf


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
