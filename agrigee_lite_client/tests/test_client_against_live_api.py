"""End-to-end test against a real, running agl_api instance.

Skipped by default — set AGRIGEE_TEST_API_URL to run it, e.g.:

    AGRIGEE_TEST_API_URL=http://192.168.3.204:8100 pytest tests/test_client_against_live_api.py

Not part of the default test run (SPECS.md §9): it depends on network access,
a live server, and (for multiple_sits) a working Earth Engine credential on
that server — none of which CI can assume.
"""

from __future__ import annotations

import os

import pytest
from shapely.geometry import Polygon

from agrigee_lite_client import AgriGEEClient

_API_URL = os.environ.get("AGRIGEE_TEST_API_URL")

pytestmark = pytest.mark.skipif(not _API_URL, reason="set AGRIGEE_TEST_API_URL to run this test")

_SAMPLE_GEOMETRY = Polygon(
    [
        (-56.421278446603054, -11.20431085146497),
        (-56.42086641797283, -11.203182131045496),
        (-56.418754238345244, -11.198938810008867),
        (-56.41853062573033, -11.198177072621217),
        (-56.38491524890757, -11.206474250296319),
        (-56.421278446603054, -11.20431085146497),
    ]
)


@pytest.fixture
def client():
    assert _API_URL is not None  # guaranteed by pytestmark's skipif whenever this actually runs
    with AgriGEEClient(_API_URL, poll_interval=1.0, poll_timeout=300) as c:
        yield c


def test_health(client: AgriGEEClient) -> None:
    assert client.health() is True


def test_list_satellites_includes_sentinel2(client: AgriGEEClient) -> None:
    assert "Sentinel2" in client.list_satellites()


def test_sits_returns_a_populated_dataframe(client: AgriGEEClient) -> None:
    df = client.get.sits(_SAMPLE_GEOMETRY, "2023-01-01", "2023-03-01", satellite="Sentinel2")

    assert df.height > 0
    assert "timestamp" in df.columns
