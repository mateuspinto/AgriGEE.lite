import ee

from agrigee_lite.vegetation_indices import VEGETATION_INDICES


class AbstractSatellite:
    """Base class for all data sources in AgriGEE.lite.

    Every satellite, radar sensor, or derived product (e.g., MapBiomas) is
    represented as an ``AbstractSatellite`` subclass.  You never instantiate
    this class directly — use one of the concrete subclasses from
    ``agrigee_lite.sat`` (e.g., ``Sentinel2``, ``Landsat8``).

    The class defines the interface that the download functions in
    ``agrigee_lite.get`` rely on: ``imageCollection`` to query GEE for the
    set of images in a date window, and ``compute`` to reduce those images
    to per-observation summary statistics ready to be saved as a
    ``pd.DataFrame``.

    Attributes
    ----------
    shortName : str
        Identifier used in cache keys and output column prefixes.
    availableBands : dict[str, str]
        Mapping from friendly name (e.g. ``"nir"``) to the original GEE
        band name (e.g. ``"SR_B5"``).
    selectedBands : list of (str, str)
        Subset of ``availableBands`` chosen at construction time, as
        ``(friendly_name, output_column_name)`` pairs.
    selectedIndices : list of (str, str, str)
        Spectral indices to compute, as
        ``(ee_expression, index_name, output_column_name)`` triples.
    pixelSize : int
        Native spatial resolution in metres.
    startDate, endDate : str
        ISO-8601 date strings bounding the sensor's valid time range.
    toDownloadSelectors : list of str
        The column names that will appear in the output DataFrame.
    """

    def __init__(self) -> None:
        self.startDate: str = ""
        self.endDate: str = ""
        self.shortName: str = "IDoNotExist"
        self.availableBands: dict[str, str] = {}
        self.selectedBands: list[tuple[str, str]] = []
        self.selectedIndices: list[tuple[str, str, str]] = []
        self.imageCollectionName: str = ""
        self.pixelSize: int = 0
        self.toDownloadSelectors: list[str] = []

    def imageCollection(self, ee_feature: ee.Feature) -> ee.ImageCollection:
        """Return the cloud-masked image collection for the given feature's geometry and date window.

        Parameters
        ----------
        ee_feature : ee.Feature
            A GEE feature with ``"s"`` (start date) and ``"e"`` (end date)
            string properties and an associated geometry.

        Returns
        -------
        ee.ImageCollection
            Filtered, cloud-masked, and band-renamed image collection.
        """
        return ee.ImageCollection()

    def compute(
        self,
        ee_feature: ee.Feature,
        subsampling_max_pixels: float,
        reducers: set[str] | None = None,
    ) -> ee.FeatureCollection:
        """Reduce the image collection to a per-date FeatureCollection of band statistics.

        This is the method called by ``download_single_sits`` and
        ``download_multiple_sits``.  Each feature in the returned collection
        corresponds to one image date and contains one value per selected
        band/index (the spatial reduction over the geometry).

        Parameters
        ----------
        ee_feature : ee.Feature
            Feature with ``"s"``/``"e"`` date properties and a geometry.
        subsampling_max_pixels : float
            Cap on the number of pixels used for spatial reduction.
            Values > 1 are treated as an absolute pixel count; values ≤ 1
            as a fraction of the total pixel count for the geometry.
        reducers : set of str or None
            Spatial aggregation functions to apply (e.g., ``{"median"}``).
            When ``None`` or a single reducer is given, per-pixel values are
            returned directly instead of aggregates.

        Returns
        -------
        ee.FeatureCollection
            Flat table of band/index values, one row per image date.
        """
        return ee.FeatureCollection()

    def log_dict(self) -> dict:
        """Return a serialisable dict describing the satellite configuration.

        Used to build cache keys so that downloads with different parameters
        (e.g., ``use_sr=True`` vs ``False``) are stored separately.
        """
        return {self.__class__.__name__: self.__dict__}

    @property
    def availableIndices(self) -> dict[str, str]:
        """Spectral indices that can be computed from the currently selected bands.

        Returns
        -------
        dict[str, str]
            Mapping from index name (e.g. ``"ndvi"``) to its GEE expression
            string.  Only indices whose required bands are all present in
            ``availableBands`` are included.
        """
        return {
            name: idx["expression"]
            for name, idx in VEGETATION_INDICES.items()
            if idx["required_bands"].issubset(self.availableBands.keys())
        }

    def __str__(self) -> str:
        return self.shortName

    def __repr__(self) -> str:
        return self.shortName


class OpticalSatellite(AbstractSatellite):
    """Base class for passive optical sensors (e.g., Sentinel-2, Landsat).

    Optical satellites capture reflected sunlight, so their imagery is
    affected by clouds and is only acquired during daylight.  Band values
    are typically surface or top-of-atmosphere reflectance in the range
    0–1 after scaling.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dateType = "optical"


class RadarSatellite(AbstractSatellite):
    """Base class for active radar sensors (e.g., Sentinel-1, PALSAR-2).

    Radar satellites emit their own microwave pulses and record the
    backscatter, so they are cloud-independent and work day and night.
    Band values are backscatter intensity in decibels (dB).
    """

    def __init__(self) -> None:
        super().__init__()
        self.dateType = "radar"


class DataSourceSatellite(AbstractSatellite):
    """Base class for derived data products that are not raw satellite imagery.

    Examples include land-use classification maps (MapBiomas) and
    satellite embeddings.  These products have their own temporal
    cadence (often annual) and their values are not reflectance values.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dateType = "dataSource"


class SingleImageSatellite(AbstractSatellite):
    """Base class for static, time-independent datasets.

    Used for products that have no time dimension, such as Digital
    Elevation Models (DEMs) and soil classification maps.  Instead of an
    ``imageCollection``, these classes expose a single ``image`` method.
    """

    def __init__(self) -> None:
        super().__init__()
        self.dateType = "singleImage"

    def image(self, ee_feature: ee.Feature) -> ee.Image:
        """Return the static GEE image for the given feature.

        Parameters
        ----------
        ee_feature : ee.Feature
            Feature providing the region of interest geometry.

        Returns
        -------
        ee.Image
            The static image clipped and prepared for reduction.
        """
        return ee.Image()
