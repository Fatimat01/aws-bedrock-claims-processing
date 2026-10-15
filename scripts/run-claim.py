"""
Run the insurance claim processing workflow.

This script resolves the claim-processing S3 bucket from the application
config, processes one claim document through Amazon Bedrock, and prints the
validated claim data, the policy-grounded summary, and inference metrics.
"""

from __future__ import annotations

import argparse
import json
import sys

from claim_poc.config import config, get_bucket_name
from claim_poc.model_invoker import ModelInvoker
from claim_poc.processor import ClaimProcessor
from claim_poc.storage import S3Storage


DEFAULT_CLAIM_ID = "CLM-2024-001"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Process one insurance claim document from S3.",
    )

    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "--claim-id",
        default=DEFAULT_CLAIM_ID,
        help=(
            "Claim ID whose document is under the configured "
            f"claims prefix (default: {DEFAULT_CLAIM_ID})."
        ),
    )

    group.add_argument(
        "--claim-key",
        help="Full S3 key of the claim document. Overrides --claim-id.",
    )

    return parser.parse_args()


def main() -> int:
    """Process a claim document and print the result."""

    args = parse_args()

    claim_key = args.claim_key or config.claim_document_key(args.claim_id)

    bucket = get_bucket_name()

    print(f"Bucket:  {bucket}", file=sys.stderr)
    print(f"Claim:   s3://{bucket}/{claim_key}", file=sys.stderr)
    print(f"Model:   {config.bedrock_model_id}", file=sys.stderr)
    print(f"Profile: {config.aws_profile}", file=sys.stderr)

    storage = S3Storage(bucket=bucket)

    model = ModelInvoker()

    processor = ClaimProcessor(
        storage=storage,
        model=model,
    )

    try:
        result = processor.process(claim_key)

    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"\nClaim processing failed: {error}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
