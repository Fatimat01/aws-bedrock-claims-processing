"""
Download the external insurance claims dataset and upload it to Amazon S3.

This script retrieves the S3 bucket name from the application config (which
falls back to Terraform output), downloads the configured public insurance
claims dataset, validates that the downloaded file is not empty, and uploads
the dataset to the claims/source prefix in the project S3 bucket.

This script is intended for local project setup and test-data acquisition.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from claim_poc.config import PROJECT_ROOT, config, get_bucket_name
from claim_poc.storage import S3Storage


DATASET_URL = (
    "https://raw.githubusercontent.com/"
    "gcc-insurance-intelligence-lab/insurance-datasets-synthetic/"
    "main/data/claims_data.csv"
)

LOCAL_FILE = PROJECT_ROOT / "data" / "claims" / "claims_data.csv"


def download_dataset() -> Path:
    """Download the source dataset to the local data directory."""

    LOCAL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Downloading dataset from:\n{DATASET_URL}")

    urllib.request.urlretrieve(
        DATASET_URL,
        LOCAL_FILE,
    )

    if not LOCAL_FILE.exists():
        raise RuntimeError("Dataset download failed.")

    if LOCAL_FILE.stat().st_size == 0:
        raise RuntimeError("Downloaded dataset is empty.")

    print(
        f"Downloaded: {LOCAL_FILE} "
        f"({LOCAL_FILE.stat().st_size:,} bytes)"
    )

    return LOCAL_FILE


def main() -> None:
    """Download the dataset and upload it to the claim bucket."""

    bucket = get_bucket_name()

    print(f"Using S3 bucket: {bucket}")

    dataset = download_dataset()

    storage = S3Storage(bucket=bucket)

    s3_uri = storage.upload_file(
        source=dataset,
        key=config.source_data_key,
    )

    print(f"Uploaded dataset to: {s3_uri}")


if __name__ == "__main__":
    main()
