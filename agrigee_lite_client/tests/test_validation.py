from __future__ import annotations

import polars as pl
import pytest

from agrigee_lite_client._validation import drop_rows_outside_satellite_range


def test_rows_inside_range_survive_untouched() -> None:
    frame = pl.DataFrame(
        {
            "id": [0, 1],
            "start_date": ["2023-01-01", "2023-02-01"],
            "end_date": ["2023-06-01", "2023-07-01"],
        }
    )

    result = drop_rows_outside_satellite_range(
        frame,
        satellite_start="2019-01-01",
        satellite_end="2050-01-01",
        start_date_column="start_date",
        end_date_column="end_date",
    )

    assert result.height == 2
    assert result.get_column("id").to_list() == [0, 1]


def test_row_before_satellite_start_is_dropped() -> None:
    frame = pl.DataFrame(
        {
            "id": [0, 1],
            "start_date": ["2023-01-01", "2000-01-01"],
            "end_date": ["2023-06-01", "2000-06-01"],
        }
    )

    result = drop_rows_outside_satellite_range(
        frame,
        satellite_start="2019-01-01",
        satellite_end="2050-01-01",
        start_date_column="start_date",
        end_date_column="end_date",
    )

    assert result.get_column("id").to_list() == [0]


def test_row_after_satellite_end_is_dropped() -> None:
    frame = pl.DataFrame(
        {
            "id": [0, 1],
            "start_date": ["2023-01-01", "2060-01-01"],
            "end_date": ["2023-06-01", "2060-06-01"],
        }
    )

    result = drop_rows_outside_satellite_range(
        frame,
        satellite_start="2019-01-01",
        satellite_end="2050-01-01",
        start_date_column="start_date",
        end_date_column="end_date",
    )

    assert result.get_column("id").to_list() == [0]


def test_zero_width_row_is_dropped_even_when_in_range() -> None:
    frame = pl.DataFrame(
        {
            "id": [0, 1],
            "start_date": ["2023-01-01", "2023-01-01"],
            "end_date": ["2023-06-01", "2023-01-01"],  # start == end
        }
    )

    result = drop_rows_outside_satellite_range(
        frame,
        satellite_start="2019-01-01",
        satellite_end="2050-01-01",
        start_date_column="start_date",
        end_date_column="end_date",
    )

    assert result.get_column("id").to_list() == [0]


def test_all_rows_outside_range_raises_value_error() -> None:
    frame = pl.DataFrame({"start_date": ["2000-01-01"], "end_date": ["2000-06-01"]})

    with pytest.raises(ValueError, match="None of the requested periods intersect"):
        drop_rows_outside_satellite_range(
            frame,
            satellite_start="2019-01-01",
            satellite_end="2050-01-01",
            start_date_column="start_date",
            end_date_column="end_date",
        )
