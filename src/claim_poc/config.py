"""
Central application configuration for the claim processing project.

This module defines shared runtime settings such as the AWS profile, AWS
region, Bedrock model identifier, and the S3 key layout used by the
scripts, services, and tests.

Values can be overridden through environment variables without changing
application source code:

    AWS_PROFILE       AWS named profile (default: neotrix)
    AWS_REGION        AWS region (default: us-east-1)
    BEDROCK_MODEL_ID  Bedrock model identifier (default: amazon.nova-pro-v1:0)
    CLAIM_BUCKET      S3 bucket name; when unset it is read from Terraform
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INFRA_DIR = PROJECT_ROOT / "infra"


@dataclass(frozen=True)
class AppConfig:
    """Shared runtime configuration for the application."""

    aws_profile: str
    aws_region: str
    bedrock_model_id: str

    # Optional explicit bucket name. When empty, get_bucket_name() falls
    # back to the Terraform output in the infra directory.
    claim_bucket: str | None = None

    # Sample claims used by the data scripts, tests, and evaluation.
    selected_claim_ids: tuple[str, ...] = (
        "CLM-2024-001",
        "CLM-2024-004",
        "CLM-2024-011",
    )

    # S3 key layout shared by the data scripts and tests.
    source_data_key: str = "claims/source/claims_data.csv"
    claim_documents_prefix: str = "claims/documents/"
    claim_expected_prefix: str = "claims/expected/"
    policies_prefix: str = "policies/"
    results_prefix: str = "results/"
    evaluations_prefix: str = "evaluations/"

    def claim_document_key(self, claim_id: str) -> str:
        """Return the S3 key of a generated claim document."""

        return f"{self.claim_documents_prefix}{claim_id}.txt"

    def claim_expected_key(self, claim_id: str) -> str:
        """Return the S3 key of a claim's expected extraction result."""

        return f"{self.claim_expected_prefix}{claim_id}.json"

    def policy_key(self, policy_number: str) -> str:
        """Return the S3 key of a policy document."""

        return f"{self.policies_prefix}{policy_number}.txt"

    def result_key(self, claim_id: str) -> str:
        """Return the S3 key of a processed claim result."""

        return f"{self.results_prefix}{claim_id}.json"

    def evaluation_key(self, run_id: str) -> str:
        """Return the S3 key of a model evaluation run."""

        return f"{self.evaluations_prefix}{run_id}.json"


config = AppConfig(
    aws_profile=os.getenv(
        "AWS_PROFILE",
        "neotrix",
    ),
    aws_region=os.getenv(
        "AWS_REGION",
        "us-east-1",
    ),
    bedrock_model_id=os.getenv(
        "BEDROCK_MODEL_ID",
        "us.amazon.nova-lite-v1:0",
    ),
    claim_bucket=os.getenv("CLAIM_BUCKET") or None,
)


def get_bucket_name() -> str:
    """
    Return the claim-processing S3 bucket name.

    The CLAIM_BUCKET environment variable takes precedence. Otherwise the
    name is read from the Terraform output in the project's infra directory,
    so this works regardless of the current working directory.

    Raises:
        RuntimeError: If Terraform does not return a bucket name.
    """

    if config.claim_bucket:
        return config.claim_bucket

    result = subprocess.run(
        [
            "terraform",
            f"-chdir={INFRA_DIR}",
            "output",
            "-raw",
            "claim_bucket_name",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    bucket = result.stdout.strip()

    if not bucket:
        raise RuntimeError(
            "Terraform did not return a claim bucket name. "
            "Run terraform apply in infra/ or set CLAIM_BUCKET."
        )

    return bucket
