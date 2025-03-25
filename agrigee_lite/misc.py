from collections import deque

import geopandas as gpd
import numpy as np


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

    gdf["centroid_x"] = gdf.geometry.centroid.x
    gdf["centroid_y"] = gdf.geometry.centroid.y

    clusters = build_quadtree_iterative(gdf, max_size=max_size)

    cluster_id = np.zeros(len(gdf), dtype=int)
    for i, cluster_indexes in enumerate(clusters):
        cluster_id[cluster_indexes] = i

    gdf["cluster_id"] = cluster_id

    return gdf
