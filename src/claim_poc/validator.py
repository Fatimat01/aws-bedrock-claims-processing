"""
Validation models for structured insurance claim data.

This module defines the schema and business-level validation rules applied
to claim information extracted by foundation models.

Validation is intentionally separated from model invocation so malformed,
incomplete, or semantically invalid model responses can be rejected before
they are used for policy retrieval, summarization, or persistence.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
)


class ClaimData(BaseModel):
    """Validated structured representation of an insurance claim."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    claim_number: str = Field(
        min_length=1,
    )

    policy_number: str = Field(
        min_length=1,
    )

    claim_date: date

    claim_type: str = Field(
        min_length=1,
    )

    claim_amount: Decimal = Field(
        gt=0,
    )

    incident_severity: str = Field(
        min_length=1,
    )

    claimant_age: int | None = Field(
        default=None,
        ge=0,
        le=120,
    )

    claimant_name: str | None = None

    @field_serializer("claim_amount", when_used="json")
    def serialize_claim_amount(
        self,
        value: Decimal,
    ) -> int | float:
        """
        Serialize the claim amount as a JSON number rather than a string.

        Decimal is kept internally so validation preserves exact precision.
        Whole amounts emit as integers and fractional amounts as floats, so
        persisted results match the numeric form used in the expected
        ground-truth files.
        """

        if value == value.to_integral_value():
            return int(value)

        return float(value)

    @field_validator("claimant_name")
    @classmethod
    def normalize_claimant_name(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize empty or explicitly unavailable claimant names."""

        if value is None:
            return None

        normalized = value.strip()

        if normalized.lower() in {
            "",
            "not provided",
            "unknown",
            "n/a",
        }:
            return None

        return normalized


class ClaimValidator:
    """Validate model-extracted claim information."""

    def validate(
        self,
        claim: dict,
    ) -> ClaimData:
        """
        Validate extracted claim data.

        Args:
            claim: Structured claim information returned by a model.

        Returns:
            A validated ClaimData instance.

        Raises:
            pydantic.ValidationError: If the claim violates the schema or
            semantic validation rules.
        """

        return ClaimData.model_validate(claim)