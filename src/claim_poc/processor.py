"""
Core insurance claim processing workflow.

This module coordinates claim retrieval, Bedrock information extraction,
claim validation, policy retrieval, summary generation, and result
persistence.
"""

from __future__ import annotations

import json

from pydantic import ValidationError

from claim_poc.config import config
from claim_poc.model_invoker import ModelInvoker
from claim_poc.policy_retriever import PolicyRetriever
from claim_poc.prompt_manager import PromptTemplateManager
from claim_poc.storage import S3Storage
from claim_poc.validator import ClaimValidator


def strip_code_fence(text: str) -> str:
    """Remove a surrounding ```json ... ``` or ``` ... ``` fence, if any."""

    text = text.strip()

    if not text.startswith("```"):
        return text

    first_newline = text.find("\n")

    if first_newline == -1 or not text.endswith("```"):
        return text

    return text[first_newline + 1 : -3].strip()


def normalize_summary(text: str) -> str:
    """
    Collapse a model summary into a single clean line of text.

    Line breaks, tabs, and repeated spaces are reduced to single spaces so
    the persisted JSON carries readable prose instead of escaped newlines.
    """

    return " ".join(text.split())


def parse_json_response(text: str) -> dict:
    """
    Parse a model response as a JSON object.

    Markdown code fences are stripped first, since models sometimes wrap
    JSON in them despite instructions not to.

    Raises:
        ValueError: If the response is not a JSON object.
    """

    text = strip_code_fence(text)

    try:
        data = json.loads(text)

    except json.JSONDecodeError as error:
        raise ValueError(
            "Bedrock returned invalid claim JSON:\n"
            f"{text}"
        ) from error

    if not isinstance(data, dict):
        raise ValueError(
            "Bedrock returned JSON that is not an object:\n"
            f"{text}"
        )

    return data


class ClaimProcessor:
    """Process an insurance claim from extraction through summary generation."""

    def __init__(
        self,
        storage: S3Storage,
        model: ModelInvoker,
    ) -> None:
        self.storage = storage
        self.model = model
        self.prompts = PromptTemplateManager()
        self.validator = ClaimValidator()
        self.policy_retriever = PolicyRetriever(storage)

    def process(
        self,
        claim_key: str,
    ) -> dict:
        """
        Process one claim document stored in Amazon S3.

        Args:
            claim_key: S3 key of the claim document to process.

        Returns:
            A result dictionary containing the validated claim, the
            policy-grounded summary, and per-step inference metrics.

        Raises:
            FileNotFoundError: If the claim or its policy is not in S3.
            ValueError: If the model response cannot be parsed or fails
                validation.
        """

        # 1. Read claim document
        document_text = self.storage.read_text(claim_key)

        # 2. Extract structured claim information
        extraction_prompt = self.prompts.get_prompt(
            "extract_claim",
            document_text=document_text,
        )

        extraction_response = self.model.invoke(
            extraction_prompt,
            temperature=0.0,
            max_tokens=1000,
        )

        raw_claim = parse_json_response(extraction_response.text)

        # 3. Validate the extracted claim
        try:
            validated = self.validator.validate(raw_claim)

        except ValidationError as error:
            raise ValueError(
                "Extracted claim failed validation:\n"
                f"{error}"
            ) from error

        # mode="json" converts Decimal and date fields to JSON-safe values.
        claim = validated.model_dump(mode="json")

        # 4. Retrieve policy information
        policy_context = self.policy_retriever.retrieve(
            validated.policy_number
        )

        # 5. Generate grounded summary
        summary_prompt = self.prompts.get_prompt(
            "generate_summary",
            claim_data=json.dumps(claim, indent=2),
            policy_context=policy_context,
        )

        summary_response = self.model.invoke(
            summary_prompt,
            temperature=0.2,
            max_tokens=500,
        )

        # 6. Build final result
        result = {
            "claim": claim,
            "summary": normalize_summary(summary_response.text),
            "metrics": {
                "model_id": self.model.model_id,
                "extraction": {
                    "input_tokens": extraction_response.input_tokens,
                    "output_tokens": extraction_response.output_tokens,
                    "latency_ms": extraction_response.latency_ms,
                },
                "summary": {
                    "input_tokens": summary_response.input_tokens,
                    "output_tokens": summary_response.output_tokens,
                    "latency_ms": summary_response.latency_ms,
                },
            },
        }

        # 7. Persist result
        result_key = config.result_key(validated.claim_number)

        self.storage.upload_json(
            data=result,
            key=result_key,
        )

        result["result_key"] = result_key

        return result
