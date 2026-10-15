"""
Unit tests for insurance policy retrieval.

These tests verify that PolicyRetriever constructs the expected S3 object key
from a policy number and delegates retrieval to the configured storage layer.

The tests use a mock storage object and do not make AWS API calls.
"""

from unittest.mock import Mock

from claim_poc.policy_retriever import PolicyRetriever


def test_policy_retriever_uses_policy_number_as_key() -> None:
    """Verify that the expected S3 policy key is requested."""

    storage = Mock()

    storage.read_text.return_value = "policy content"

    retriever = PolicyRetriever(
        storage=storage,
    )

    result = retriever.retrieve(
        "POL-A-1001"
    )

    storage.read_text.assert_called_once_with(
        "policies/POL-A-1001.txt"
    )

    assert result == "policy content"