"""
Policy retrieval utilities for the claim processing RAG workflow.

This module retrieves policy documents from Amazon S3 using the policy number
extracted from a claim.

The initial implementation uses deterministic S3 key lookup. This keeps the
first RAG implementation simple while preserving an interface that can later
be replaced with Amazon Bedrock Knowledge Bases or another retrieval system.
"""

from __future__ import annotations

from claim_poc.config import config
from claim_poc.storage import S3Storage


class PolicyRetriever:
    """Retrieve insurance policy context from Amazon S3."""

    def __init__(
        self,
        storage: S3Storage,
    ) -> None:
        self.storage = storage

    def retrieve(
        self,
        policy_number: str,
    ) -> str:
        """
        Retrieve the policy document associated with a policy number.

        Args:
            policy_number: Policy identifier extracted from the claim.

        Returns:
            The policy document text.

        Raises:
            FileNotFoundError: If the corresponding policy document does not
            exist in Amazon S3.
        """

        key = config.policy_key(policy_number)

        return self.storage.read_text(key)