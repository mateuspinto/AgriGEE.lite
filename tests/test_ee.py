import ee


def test_ee_authenticate() -> None:
    ee.Initialize(opt_url="https://earthengine-highvolume.googleapis.com", project="ee-paulagibrim")
    assert ee.data._initialized
