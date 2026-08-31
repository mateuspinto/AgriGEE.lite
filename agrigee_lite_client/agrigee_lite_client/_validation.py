"""Client-side mirror of the server's date/satellite-range validation.

Mirrors agrigee_lite/get/sits.py: sanitize_and_prepare_input_gdf — the masking,
dropping, and logging around rows whose [start_date, end_date) doesn't
intersect the satellite's temporal range. Applying this before ever opening a
connection means an impossible request fails immediately and locally instead
of round-tripping to the server first (and, for `multiple_sits`, before
uploading a GeoParquet full of rows the server would just drop anyway).

H3 clustering (also part of the server function) is intentionally not
mirrored here — it's a chunking/performance detail for the server's own
download scheduling, not part of request validity.
"""

from __future__ import annotations

import logging
from datetime import datetime

import polars as pl

logger = logging.getLogger(__name__)


def _as_datetime_expr(frame: pl.DataFrame, column: str) -> pl.Expr:
    """pl.col(column).cast(pl.Datetime) silently turns every value into null
    for a String column as of polars 1.43 (the cast used to parse; that
    behavior is deprecated/gone, str.to_datetime() is the replacement) — a
    trap the server itself never hits because it always converts date
    columns with pandas.to_datetime() before this kind of check runs, but
    this client hands raw ISO strings straight from sits()/multiple_sits()
    callers.
    """
    if frame.schema[column] == pl.Utf8:
        return pl.col(column).str.to_datetime(strict=False)
    return pl.col(column).cast(pl.Datetime, strict=False)


def drop_rows_outside_satellite_range(
    frame: pl.DataFrame,
    *,
    satellite_start: str,
    satellite_end: str,
    start_date_column: str,
    end_date_column: str,
) -> pl.DataFrame:
    """Drop rows that don't intersect [satellite_start, satellite_end).

    Raises ValueError if *none* of the rows intersect (mirrors the server's
    behavior exactly). A row with start_date == end_date is dropped too —
    Earth Engine's date filter treats the end date as exclusive, so a
    zero-width window is never valid.
    """
    working = frame.with_columns(
        _as_datetime_expr(frame, start_date_column).alias(start_date_column),
        _as_datetime_expr(frame, end_date_column).alias(end_date_column),
    )

    sat_start = datetime.fromisoformat(satellite_start)
    sat_end = datetime.fromisoformat(satellite_end)

    working = working.with_columns(
        (
            (pl.col(end_date_column) < pl.lit(sat_start)) | (pl.col(start_date_column) > pl.lit(sat_end))
        ).alias("_mask_no_intersection"),
        (pl.col(start_date_column) == pl.col(end_date_column)).alias("_mask_zero_width"),
    )

    height = working.height
    count_none = int(working.get_column("_mask_no_intersection").sum())
    count_zero_width = int(working.get_column("_mask_zero_width").sum())

    pct_none = 100 * count_none / height
    if pct_none > 0:
        logger.warning("%.2f%% of the data do not intersect the satellite period.", pct_none)

    if count_zero_width > 0:
        pct_zero_width = 100 * count_zero_width / height
        logger.warning(
            "%d row(s) (%.2f%%) have start_date == end_date and were dropped: Earth Engine's date "
            "filter treats the end date as exclusive, so a zero-width range is not supported.",
            count_zero_width,
            pct_zero_width,
        )

    if count_none == height:
        msg = f"None of the requested periods intersect the satellite's temporal range ({satellite_start} to {satellite_end})."
        raise ValueError(msg)

    drop_mask = pl.col("_mask_no_intersection") | pl.col("_mask_zero_width")
    return working.filter(~drop_mask).drop("_mask_no_intersection", "_mask_zero_width")
