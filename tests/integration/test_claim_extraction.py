"""
Integration tests for insurance claim extraction with Amazon Bedrock.

These tests retrieve a generated claim document and its expected structured
values from Amazon S3, invoke Amazon Nova Pro through the reusable
ModelInvoker, and compare the extracted model response against the known
ground-truth values.

The tests make real AWS API calls and may incur a small Amazon Bedrock cost.
"""

from __future__ import annotations

import json

from claim_poc.config import config, get_bucket_name
from claim_poc.model_invoker import ModelInvoker
from claim_poc.prompt_manager import PromptTemplateManager
from claim_poc.storage import S3Storage


CLAIM_ID = "CLM-2024-001"

CLAIM_KEY = config.claim_document_key(CLAIM_ID)
EXPECTED_KEY = config.claim_expected_key(CLAIM_ID)


def test_configured_model_extracts_claim_correctly() -> None:
    """Verify that the configured model extracts the expected values."""

    bucket = get_bucket_name()

    storage = S3Storage(bucket=bucket)

    model = ModelInvoker()

    prompts = PromptTemplateManager()

    document_text = storage.read_text(
        CLAIM_KEY
    )

    expected = json.loads(
        storage.read_text(EXPECTED_KEY)
    )

    prompt = prompts.get_prompt(
        "extract_claim",
        document_text=document_text,
    )

    response = model.invoke(
        prompt,
        temperature=0.0,
        max_tokens=1000,
    )

    actual = json.loads(
        response.text
    )

    assert actual == expected