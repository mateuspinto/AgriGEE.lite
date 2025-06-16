import concurrent.futures
import hashlib
import inspect
import warnings
from collections import deque
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import ParamSpec, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon, Point, Polygon
from topojson import Topology
from tqdm.std import tqdm


def build_quadtree_iterative(gdf: gpd.GeoDataFrame, max_size: int = 1000) -> list[int]:
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


def build_quadtree(gdf: gpd.GeoDataFrame, max_size: int = 1000, depth: int = 0) -> list[int]:
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
    simplified = simplify_gdf(cluster)
    return cluster_id, simplified.geometry


def quadtree_clustering(
    gdf: gpd.GeoDataFrame,
    max_size: int = 1_000,
) -> gpd.GeoDataFrame:
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


def create_gdf_hash(gdf: gpd.GeoDataFrame) -> str:
    gdf_copy = gdf[["geometry", "start_date", "end_date"]].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf_copy["centroid_x"] = gdf_copy.geometry.centroid.x
        gdf_copy["centroid_y"] = gdf_copy.geometry.centroid.y

    gdf_copy = gdf_copy.drop(columns=["geometry"])

    hash_values = pd.util.hash_pandas_object(gdf_copy).values
    return hashlib.sha1(hash_values).hexdigest()  # type: ignore  # noqa: PGH003, S324


P = ParamSpec("P")
R = TypeVar("R")


def cached(func: Callable[P, R]) -> Callable[P, R]:
    cached_func = lru_cache()(func)

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return cached_func(*args, **kwargs)  # type: ignore  # noqa: PGH003

    return wrapper


def remove_underscore_in_df(df: pd.DataFrame | gpd.GeoDataFrame) -> None:
    df.columns = [column.split("_", 1)[1] for column in df.columns.tolist()]


def compute_index_from_df(df: pd.DataFrame, np_function: Callable) -> np.ndarray:
    sig = inspect.signature(np_function)
    kwargs = {}

    index_name = str(np_function.__name__).split("np_")[1]

    if index_name in df.columns.tolist():
        return df[index_name].to_numpy()

    for param_name, param in sig.parameters.items():
        if param_name in df.columns:
            kwargs[param_name] = df[param_name].values
        else:
            if param.default is not inspect._empty:
                kwargs[param_name] = param.default
            else:
                raise ValueError(  # noqa: TRY003
                    f"DataFrame is missing a column '{param_name}', "
                    f"required by {np_function.__name__}, and there's no default."
                )

    return np_function(**kwargs)


def add_indexnum_column(df: pd.DataFrame) -> None:
    if "00_indexnum" not in df.columns:
        if not (df.index.to_numpy() == np.arange(len(df))).all():
            raise ValueError(  # noqa: TRY003
                "The index must be sequential from 0 to N-1. To do this, use gdf.reset_index(drop=True) before executing this function."
            )
        df["00_indexnum"] = range(len(df))


def log_dict_function_call_summary(ignore: list[str] | None = None) -> dict[str, dict[str, str]]:
    frame = inspect.currentframe().f_back
    func_name = frame.f_code.co_name
    args, _, _, values = inspect.getargvalues(frame)
    ignore = ignore or []
    args_dict = {str(arg): str(values[arg]) for arg in args if arg not in ignore}
    return {func_name: args_dict}


def create_grid_centroids_numpy(geometry: Polygon | MultiPolygon, n_cells=10) -> np.ndarray:
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


def generate_grid_random_points_from_gdf(gdf: gpd.GeoDataFrame, num_points_per_geometry=10) -> gpd.GeoDataFrame:
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

    return points_gdf
