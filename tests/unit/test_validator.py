"""
Unit tests for insurance claim validation.

These tests verify that structured claim data is accepted when valid and
rejected when required fields, numeric ranges, or schema constraints are
violated.

The tests do not call AWS services and can run entirely locally.
"""

from __future__ import annotations

import json
from decimal import Decimal

import pytest
from pydantic import ValidationError

from claim_poc.validator import ClaimValidator


def valid_claim() -> dict:
    """Return a representative valid claim payload."""

    return {
        "claim_number": "CLM-2024-001",
        "policy_number": "POL-A-1001",
        "claim_date": "2024-01-15",
        "claim_type": "Auto Collision",
        "claim_amount": 12500,
        "incident_severity": "Moderate",
        "claimant_age": 34,
        "claimant_name": None,
    }


def test_valid_claim_is_accepted() -> None:
    """Verify that a valid claim passes validation."""

    validator = ClaimValidator()

    result = validator.validate(
        valid_claim()
    )

    assert result.claim_number == "CLM-2024-001"
    assert result.policy_number == "POL-A-1001"
    assert result.claim_amount == 12500
    assert result.claimant_age == 34
    assert result.claimant_name is None


def test_negative_claim_amount_is_rejected() -> None:
    """Verify that claim amounts must be greater than zero."""

    claim = valid_claim()
    claim["claim_amount"] = -500

    validator = ClaimValidator()

    with pytest.raises(ValidationError):
        validator.validate(claim)


def test_unreasonable_claimant_age_is_rejected() -> None:
    """Verify that claimant age must remain within an expected range."""

    claim = valid_claim()
    claim["claimant_age"] = 250

    validator = ClaimValidator()

    with pytest.raises(ValidationError):
        validator.validate(claim)


def test_missing_required_field_is_rejected() -> None:
    """Verify that required claim fields cannot be omitted."""

    claim = valid_claim()

    del claim["policy_number"]

    validator = ClaimValidator()

    with pytest.raises(ValidationError):
        validator.validate(claim)


def test_unexpected_field_is_rejected() -> None:
    """Verify that model-generated fields outside the schema are rejected."""

    claim = valid_claim()

    claim["approval_recommendation"] = "approve"

    validator = ClaimValidator()

    with pytest.raises(ValidationError):
        validator.validate(claim)


def test_not_provided_claimant_name_becomes_null() -> None:
    """Verify that unavailable claimant names are normalized to None."""

    claim = valid_claim()

    claim["claimant_name"] = "Not provided"

    validator = ClaimValidator()

    result = validator.validate(claim)

    assert result.claimant_name is None


def test_claim_amount_serializes_as_json_number() -> None:
    """Verify that the claim amount is persisted as a number, not a string."""

    validator = ClaimValidator()

    result = validator.validate(
        valid_claim()
    )

    dumped = result.model_dump(mode="json")

    assert dumped["claim_amount"] == 12500
    assert not isinstance(dumped["claim_amount"], str)
    assert '"claim_amount": "' not in json.dumps(dumped)


def test_fractional_claim_amount_keeps_cents() -> None:
    """Verify that fractional amounts serialize as floats without loss."""

    claim = valid_claim()
    claim["claim_amount"] = "1234.56"

    validator = ClaimValidator()

    result = validator.validate(claim)

    assert result.claim_amount == Decimal("1234.56")
    assert result.model_dump(mode="json")["claim_amount"] == 1234.56