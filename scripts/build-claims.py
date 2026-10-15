"""
Build sample claim documents from the source insurance claims dataset.

This script reads the source claims CSV from Amazon S3, selects a small set of
claim records, transforms them into unstructured claim documents, and stores
both the generated documents and their expected structured values in S3.

The expected JSON files serve as ground truth for later model evaluation.
"""

import csv
import io

from claim_poc.config import config, get_bucket_name
from claim_poc.storage import S3Storage


SELECTED_CLAIMS = set(config.selected_claim_ids)


def build_claim_document(row: dict[str, str]) -> str:
    """Render one CSV row as an unstructured claim submission document."""

    return f"""
INSURANCE CLAIM SUBMISSION

Claim Number: {row["claim_id"]}
Policy Number: {row["policy_id"]}

Claim Date: {row["claim_date"]}
Claim Type: {row["claim_type"]}

Claim Amount: ${float(row["claim_amount"]):,.2f}

Incident Severity: {row["incident_severity"]}

Claimant Age: {row["claimant_age"]}

Claimant Name: Not provided

Incident Description:
The submitted claim concerns a {row["claim_type"].lower()}
incident classified as {row["incident_severity"].lower()}
severity.

The claimant is requesting reimbursement in the amount of
${float(row["claim_amount"]):,.2f}.

This document represents the initial claim submission.
""".strip()


def build_expected_result(
    row: dict[str, str],
) -> dict:
    """Return the ground-truth extraction result for one CSV row."""

    return {
        "claim_number": row["claim_id"],
        "policy_number": row["policy_id"],
        "claim_date": row["claim_date"],
        "claim_type": row["claim_type"],
        "claim_amount": float(row["claim_amount"]),
        "incident_severity": row["incident_severity"],
        "claimant_age": int(row["claimant_age"]),
        "claimant_name": None,
    }


def main() -> None:
    """Build claim documents and expected results, then upload them."""

    bucket = get_bucket_name()

    storage = S3Storage(bucket=bucket)

    print(
        f"Reading source data from "
        f"s3://{bucket}/{config.source_data_key}"
    )

    csv_text = storage.read_text(
        config.source_data_key
    )

    reader = csv.DictReader(
        io.StringIO(csv_text)
    )

    created = 0

    for row in reader:
        claim_id = row["claim_id"]

        if claim_id not in SELECTED_CLAIMS:
            continue

        document = build_claim_document(row)

        expected = build_expected_result(row)

        document_key = config.claim_document_key(claim_id)

        expected_key = config.claim_expected_key(claim_id)

        storage.upload_text(
            text=document,
            key=document_key,
        )

        storage.upload_json(
            data=expected,
            key=expected_key,
        )

        print(
            f"Created {document_key}"
        )

        print(
            f"Created {expected_key}"
        )

        created += 1

    if created != len(SELECTED_CLAIMS):
        raise RuntimeError(
            f"Expected to create "
            f"{len(SELECTED_CLAIMS)} claims, "
            f"but created {created}."
        )

    print(
        f"\nCreated {created} claim documents."
    )


if __name__ == "__main__":
    main()