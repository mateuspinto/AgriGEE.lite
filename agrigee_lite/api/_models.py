"""Pydantic request / response models for the AgriGEE.lite API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agrigee_lite.config import ASYNC_MAX_PARALLEL_DOWNLOADS, ASYNC_MAX_RETRIES_PER_CHUNK, SITS_CHUNKSIZE
from agrigee_lite.api._jobs import JobStatus, JobType

# ---------------------------------------------------------------------------
# Sample data — extracted from agrigee_lite.get_sample_gdf().iloc[0]
# Used as OpenAPI examples so Swagger UI is ready to test immediately.
# ---------------------------------------------------------------------------

_SAMPLE_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [
        [
            [-56.421278446603054, -11.20431085146497],
            [-56.42086641797283, -11.203182131045496],
            [-56.418754238345244, -11.198938810008867],
            [-56.41853062573033, -11.198177072621217],
            [-56.41816897285581, -11.198243694391246],
            [-56.38491524890757, -11.206474250296319],
            [-56.40228720556215, -11.210026776096111],
            [-56.409204401016275, -11.211470180586069],
            [-56.412799081653965, -11.214714718181126],
            [-56.42063419748722, -11.206564164779705],
            [-56.42136812565055, -11.205220784755152],
            [-56.421278446603054, -11.20431085146497],
        ]
    ],
}

_SAMPLE_FEATURE_COLLECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "properties": {"start_date": "2017-10-01", "end_date": "2018-10-01"},
            "geometry": _SAMPLE_GEOMETRY,
        },
        {
            "type": "Feature",
            "properties": {"start_date": "2017-10-01", "end_date": "2018-10-01"},
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [
                        [-56.43, -11.21],
                        [-56.42, -11.21],
                        [-56.42, -11.20],
                        [-56.43, -11.20],
                        [-56.43, -11.21],
                    ]
                ],
            },
        },
    ],
}


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------


class SatelliteSpec(BaseModel):
    """Identifies a satellite class and its constructor kwargs."""

    name: str = Field(..., examples=["Sentinel2", "Landsat8", "Sentinel1GRD"])
    params: dict[str, Any] = Field(default_factory=dict)


class GeoJSONGeometry(BaseModel):
    """Any valid GeoJSON geometry object."""

    type: str
    coordinates: Any


class GeoJSONFeatureCollection(BaseModel):
    """GeoJSON FeatureCollection used to pass a GeoDataFrame over HTTP."""

    type: Literal["FeatureCollection"]
    features: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# /images
# ---------------------------------------------------------------------------


class ImagesRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "geometry": _SAMPLE_GEOMETRY,
                "start_date": "2017-10-01",
                "end_date": "2018-10-01",
                "satellite": {"name": "Sentinel2", "params": {}},
                "invalid_images_threshold": 0.5,
                "max_parallel_downloads": ASYNC_MAX_PARALLEL_DOWNLOADS,
                "force_redownload": False,
                "image_indices": [0],
            }
        }
    )

    geometry: GeoJSONGeometry
    start_date: str
    end_date: str
    satellite: SatelliteSpec
    invalid_images_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_parallel_downloads: int = Field(ASYNC_MAX_PARALLEL_DOWNLOADS, ge=1)
    force_redownload: bool = False
    image_indices: list[int] | None = None


class ImagesResult(BaseModel):
    dates: list[str]
    cache_dir: str


# ---------------------------------------------------------------------------
# /sits
# ---------------------------------------------------------------------------


class SitsRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "geometry": _SAMPLE_GEOMETRY,
                "start_date": "2017-10-01",
                "end_date": "2018-10-01",
                "satellite": {"name": "Sentinel2", "params": {}},
                "reducers": None,
                "subsampling_max_pixels": 1000,
            }
        }
    )

    geometry: GeoJSONGeometry
    start_date: str
    end_date: str
    satellite: SatelliteSpec
    reducers: list[str] | None = None
    subsampling_max_pixels: float = 1_000


class MultipleSitsRequest(BaseModel):
    """
    The GeoDataFrame is encoded as a GeoJSON FeatureCollection.

    Each Feature's properties must include ``start_date`` and ``end_date``
    (ISO 8601 strings) and optionally the original-index column.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "feature_collection": _SAMPLE_FEATURE_COLLECTION,
                "satellite": {"name": "Sentinel2", "params": {}},
                "reducers": None,
                "start_date_column": "start_date",
                "end_date_column": "end_date",
                "original_index_column": "original_index",
                "subsampling_max_pixels": 1000,
                "chunksize": SITS_CHUNKSIZE,
                "max_parallel_downloads": ASYNC_MAX_PARALLEL_DOWNLOADS,
                "max_retries_per_chunk": ASYNC_MAX_RETRIES_PER_CHUNK,
                "force_redownload": False,
            }
        }
    )

    feature_collection: GeoJSONFeatureCollection
    satellite: SatelliteSpec
    reducers: list[str] | None = None
    start_date_column: str = "start_date"
    end_date_column: str = "end_date"
    original_index_column: str = "original_index"
    subsampling_max_pixels: float = 1_000
    chunksize: int = SITS_CHUNKSIZE
    max_parallel_downloads: int = ASYNC_MAX_PARALLEL_DOWNLOADS
    max_retries_per_chunk: int = ASYNC_MAX_RETRIES_PER_CHUNK
    force_redownload: bool = False


# ---------------------------------------------------------------------------
# Job responses
# ---------------------------------------------------------------------------


class JobResponse(BaseModel):
    id: str
    type: JobType | None = None
    status: JobStatus
    result: Any = None
    error: str | None = None
