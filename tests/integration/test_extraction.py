"""
Test structured claim extraction using Amazon Bedrock.

This script reads one generated claim document from Amazon S3, builds the
claim-extraction prompt, invokes Amazon Nova Pro through the reusable
ModelInvoker, and parses the model response as JSON.

The script is used as an integration checkpoint before extraction logic is
moved into the main ClaimProcessor service.
"""

from __future__ import annotations

import json

from claim_poc.config import config, get_bucket_name
from claim_poc.model_invoker import ModelInvoker
from claim_poc.prompt_manager import PromptTemplateManager
from claim_poc.storage import S3Storage


CLAIM_ID = "CLM-2024-001"

CLAIM_KEY = config.claim_document_key(CLAIM_ID)


def main() -> None:
    """Run one end-to-end claim extraction test."""

    bucket = get_bucket_name()

    storage = S3Storage(bucket=bucket)

    prompts = PromptTemplateManager()

    model = ModelInvoker()

    print(
        f"Reading claim from "
        f"s3://{bucket}/{CLAIM_KEY}"
    )

    document_text = storage.read_text(
        CLAIM_KEY
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

    try:
        extracted = json.loads(
            response.text
        )

    except json.JSONDecodeError as error:
        print("\nRaw model response:")
        print(response.text)

        raise RuntimeError(
            "Model did not return valid JSON."
        ) from error

    print("\nExtracted claim:")
    print(
        json.dumps(
            extracted,
            indent=2,
        )
    )

    print("\nInference metrics:")
    print(
        f"Model:         {response.model_id}"
    )
    print(
        f"Input tokens:  {response.input_tokens}"
    )
    print(
        f"Output tokens: {response.output_tokens}"
    )
    print(
        f"Total tokens:  {response.total_tokens}"
    )
    print(
        f"Latency:       {response.latency_ms} ms"
    )


if __name__ == "__main__":
    main()