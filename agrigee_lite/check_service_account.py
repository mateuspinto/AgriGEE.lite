import argparse
import json

import ee


def main():
    parser = argparse.ArgumentParser(description="Check GEE Service Account Authentication")
    parser.add_argument("service_account_key", type=str, help="Path to the service account JSON key file")
    args = parser.parse_args()
    service_account_key = args.service_account_key

    credentials = ee.ServiceAccountCredentials(service_account_key, service_account_key)
    ee.Initialize(credentials)

    ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").first().getInfo()

    with open(service_account_key) as f:
        key_data = json.load(f)
        print(
            f"Earth Engine initialized successfully using AgriGEE.lite with service account. Project: {key_data.get('project_id', 'Unknown')}, Email: {key_data.get('client_email', 'Unknown')}."
        )


if __name__ == "__main__":
    main()
