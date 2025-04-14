import hashlib
import warnings
from collections import deque
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import ParamSpec, TypeVar

import geopandas as gpd
import numpy as np
import pandas as pd


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


def quadtree_clustering(gdf: gpd.GeoDataFrame, max_size: int = 1000) -> gpd.GeoDataFrame:
    gdf = gdf.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf["centroid_x"] = gdf.geometry.centroid.x
        gdf["centroid_y"] = gdf.geometry.centroid.y

    clusters = build_quadtree_iterative(gdf, max_size=max_size)

    cluster_id = np.zeros(len(gdf), dtype=int)
    for i, cluster_indexes in enumerate(clusters):
        cluster_id[cluster_indexes] = i

    gdf["cluster_id"] = cluster_id

    return gdf


def create_gdf_hash(gdf: gpd.GeoDataFrame) -> str:
    gdf_copy = gdf[["geometry", "start_date", "end_date"]].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gdf_copy["centroid_x"] = gdf_copy.geometry.centroid.x
        gdf_copy["centroid_y"] = gdf_copy.geometry.centroid.y

    gdf_copy = gdf_copy.drop(columns=["geometry"])

    hash_values = pd.util.hash_pandas_object(gdf_copy).values
    return hashlib.sha1(hash_values).hexdigest()  # type: ignore # noqa: S324


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
    df_wide.columns = [f"{prefix}_{col}_{seq}" for col, seq in df_wide.columns]

    df_wide = df_wide.fillna(0).copy()

    for col in df_wide.columns:
        for orig_col in original_dtypes:
            if col.startswith(f"{prefix}_{orig_col}_"):
                df_wide[col] = df_wide[col].astype(original_dtypes[orig_col])
                break

    return df_wide.reset_index()
