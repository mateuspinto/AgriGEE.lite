import ee
import pytest


@pytest.fixture(scope="session", autouse=True)
def test_ee_authenticate() -> None:
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project="ee-paulagibrim")
    assert ee.data._initialized
