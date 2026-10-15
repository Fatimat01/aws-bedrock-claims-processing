"""
Build sample insurance policy documents and upload them to Amazon S3.

Policy numbers are read from the generated claims rather than hardcoded, so
every claim the pipeline processes always has a matching policy document.
A policy template is selected by claim type and rendered with the real
policy number from the claim.

The policies are used by the retrieval step, where policy context is pulled
from S3 and injected into a Bedrock prompt for grounded claim summaries.
"""

from __future__ import annotations

import json

from claim_poc.config import config, get_bucket_name
from claim_poc.storage import S3Storage


AUTO_COLLISION_POLICY = """
AUTO INSURANCE POLICY

Policy Number: {policy_number}
Policy Type: Personal Auto Insurance

Coverage:
- Collision coverage limit: $50,000
- Property damage liability limit: $25,000
- Bodily injury liability limit: $100,000
- Collision deductible: $1,000

Requirements:
- Claims must be reported within 30 days of the incident.
- A police report should be provided when available.
- Repair estimates are required for collision claims.

Exclusions:
- Intentional damage
- Racing or competitive driving
- Commercial use unless separately endorsed
""".strip()


AUTO_THEFT_POLICY = """
AUTO INSURANCE POLICY

Policy Number: {policy_number}
Policy Type: Personal Auto Insurance

Coverage:
- Comprehensive coverage limit: Actual Cash Value
- Theft is covered under comprehensive coverage.
- Comprehensive deductible: $750
- Rental reimbursement limit: $900 per claim

Requirements:
- Vehicle theft must be reported to law enforcement.
- A police report number must be supplied.
- Proof of ownership may be requested.

Exclusions:
- Fraudulent claims
- Intentional loss
- Theft of personal property left inside the vehicle
- Commercial use unless endorsed
""".strip()


HOME_FIRE_POLICY = """
HOME INSURANCE POLICY

Policy Number: {policy_number}
Policy Type: Homeowners Insurance

Coverage:
- Dwelling coverage limit: $350,000
- Personal property coverage limit: $100,000
- Fire damage is covered subject to policy terms.
- Additional living expenses limit: $30,000
- Deductible: $2,500

Requirements:
- Loss must be reported promptly.
- Photographs and repair estimates should be supplied.
- Fire claims may require a fire department report.

Exclusions:
- Intentional loss
- Normal wear and tear
- Unreported commercial activity
""".strip()


# Claim type to policy template. Keys are matched case-insensitively.
POLICY_TEMPLATES = {
    "auto collision": AUTO_COLLISION_POLICY,
    "auto theft": AUTO_THEFT_POLICY,
    "home fire": HOME_FIRE_POLICY,
}


def select_template(claim_type: str) -> str:
    """
    Return the policy template matching a claim type.

    Raises:
        KeyError: If no template covers the claim type.
    """

    key = claim_type.strip().lower()

    if key in POLICY_TEMPLATES:
        return POLICY_TEMPLATES[key]

    raise KeyError(
        f"No policy template for claim type {claim_type!r}. "
        f"Known types: {sorted(POLICY_TEMPLATES)}"
    )


def main() -> None:
    """Upload one policy document per policy referenced by a claim."""

    bucket = get_bucket_name()

    storage = S3Storage(bucket=bucket)

    uploaded = 0

    for claim_id in config.selected_claim_ids:
        expected = json.loads(
            storage.read_text(config.claim_expected_key(claim_id))
        )

        policy_number = expected["policy_number"]
        claim_type = expected["claim_type"]

        template = select_template(claim_type)

        policy_text = template.format(policy_number=policy_number)

        uri = storage.upload_text(
            text=policy_text,
            key=config.policy_key(policy_number),
        )

        print(f"{claim_id} ({claim_type}) -> {uri}")

        uploaded += 1

    print(f"\nUploaded {uploaded} policy documents.")


if __name__ == "__main__":
    main()
