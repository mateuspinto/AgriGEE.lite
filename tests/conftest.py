import pytest

from agrigee_lite.ee_utils import ee_quick_start


@pytest.fixture(scope="session", autouse=True)
def ee_auth() -> None:
    ee_quick_start()
    # if not ee_is_authenticated():
    #     pytest.exit("Earth Engine not initialized. Set the GEE_KEY environment variable.", returncode=1)
