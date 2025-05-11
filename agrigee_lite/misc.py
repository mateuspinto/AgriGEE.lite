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


def long_to_wide_dataframe(df: pd.DataFrame, prefix: str = "", group_col: str = "indexnum") -> pd.DataFrame:
    original_dtypes = df.drop(columns=[group_col]).dtypes.to_dict()
    df["__seq__"] = df.groupby(group_col).cumcount()
    df_wide = df.pivot(index=group_col, columns="__seq__")
    df_wide.columns = [f"{prefix}_{col}_{seq}" for col, seq in df_wide.columns]  # type: ignore  # noqa: PGH003

    df_wide = df_wide.fillna(0).copy()

    for col in df_wide.columns:
        for orig_col in original_dtypes:
            if col.startswith(f"{prefix}_{orig_col}_"):
                df_wide[col] = df_wide[col].astype(original_dtypes[orig_col])
                break

    obs_count = pd.DataFrame(df.groupby(group_col).size().rename(f"{prefix}_observations"))
    df_wide = obs_count.join(df_wide)

    return df_wide.reset_index()


def wide_to_long_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["indexnum"] = range(len(df))
    df_long = df.melt(id_vars=["indexnum"], var_name="band_time", value_name="value")

    df_long[["prefix", "band", "idx"]] = df_long["band_time"].str.extract(r"([^_]+)_(\w+)_(\d+)")

    df_long["idx"] = df_long["idx"].astype(int)
    df_long["value"] = pd.to_numeric(df_long["value"], errors="coerce")

    df_pivot = df_long.pivot(index=["indexnum", "idx"], columns="band", values="value").reset_index()
    df_pivot.sort_values(by=["indexnum", "idx"], inplace=True)
    df_pivot = df_pivot.drop(columns=["idx"])
    df_pivot.columns.name = None

    return df_pivot


def compute_index_from_df(df: pd.DataFrame, np_function: Callable) -> np.ndarray:
    sig = inspect.signature(np_function)
    kwargs = {}

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


def reconstruct_df_with_indexnum(whole_result_df: pd.DataFrame, N: int) -> pd.DataFrame:
    if "indexnum" not in whole_result_df.columns:
        raise ValueError("'indexnum' column is required")  # noqa: TRY003

    all_indexes = pd.DataFrame({"indexnum": range(N)})

    merged = all_indexes.merge(whole_result_df, on="indexnum", how="left")

    filled = merged.fillna(0)

    return filled.sort_values(by="indexnum", kind="stable").reset_index(drop=True).drop(columns=["indexnum"])


def reduce_results_dataframe_size(whole_results_df: pd.DataFrame) -> pd.DataFrame:
    int_columns = list(
        filter(
            lambda x: x.split("_", 1)[1].split("_", 1)[0] in {"doy", "class", "year", "observations"},
            whole_results_df.columns.tolist(),
        )
    )
    float_columns = list(
        filter(
            lambda x: x.split("_", 1)[1].split("_", 1)[0]
            not in {"doy", "class", "year", "fyear", "timestamp", "observations"},
            whole_results_df.columns.tolist(),
        )
    )

    if "fyear" in whole_results_df.columns.tolist():
        whole_results_df["fyear"] = whole_results_df["fyear"].astype(np.float32)

    whole_results_df[int_columns] = whole_results_df[int_columns].astype(np.uint16)
    whole_results_df[float_columns] = whole_results_df[float_columns].astype(np.float16)

    return whole_results_df
