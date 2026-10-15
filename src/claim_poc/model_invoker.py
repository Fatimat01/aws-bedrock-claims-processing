"""
Amazon Bedrock model invocation utilities.

This module provides a reusable interface for invoking foundation models
through the Amazon Bedrock Runtime Converse API.

The model invocation logic is kept separate from claim-processing logic so
that different supported Bedrock models can be evaluated without changing
the rest of the application.
"""

from __future__ import annotations

from dataclasses import dataclass

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from claim_poc.config import config


@dataclass
class ModelResponse:
    """Normalized response returned by the Bedrock model invoker."""

    text: str
    model_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    latency_ms: int


class ModelInvoker:
    """Invoke supported Amazon Bedrock models through the Converse API."""

    def __init__(
        self,
        model_id: str | None = None,
        region: str | None = None,
        profile: str | None = None,
    ) -> None:
        """
        Create a Bedrock Runtime client for the given model.

        Args:
            model_id: Bedrock model ID. Defaults to config.bedrock_model_id.
            region: AWS region. Defaults to config.aws_region.
            profile: AWS named profile. Defaults to config.aws_profile.
        """

        self.model_id = model_id or config.bedrock_model_id

        session = boto3.Session(
            profile_name=profile or config.aws_profile,
            region_name=region or config.aws_region,
        )

        self.client = session.client(
            "bedrock-runtime",
            config=Config(
                connect_timeout=30,
                read_timeout=300,
                # Adaptive mode adds client-side rate limiting, which keeps
                # batch evaluation runs from exhausting the account's
                # Bedrock throughput quota.
                retries={
                    "max_attempts": 5,
                    "mode": "adaptive",
                },
            ),
        )

    def invoke(
        self,
        prompt: str,
        *,
        temperature: float | None = 0.0,
        max_tokens: int = 1024,
    ) -> ModelResponse:
        """
        Send a text prompt to the configured Bedrock model.

        Args:
            prompt: User prompt sent to the model.
            temperature: Sampling temperature used for generation. Pass None
                to omit it, which is required by newer reasoning models that
                reject sampling parameters.
            max_tokens: Maximum number of output tokens.

        Returns:
            A normalized ModelResponse containing the generated text and
            basic inference metrics.

        Raises:
            RuntimeError: If Bedrock cannot invoke the configured model.
        """

        inference_config: dict[str, object] = {"maxTokens": max_tokens}

        if temperature is not None:
            inference_config["temperature"] = temperature

        try:
            response = self.client.converse(
                modelId=self.model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt,
                            }
                        ],
                    }
                ],
                inferenceConfig=inference_config,
            )

        except ClientError as error:
            error_code = error.response["Error"]["Code"]
            error_message = error.response["Error"]["Message"]

            raise RuntimeError(
                f"Bedrock invocation failed "
                f"({error_code}): {error_message}"
            ) from error

        content = response["output"]["message"]["content"]

        text = next(
            (
                item["text"]
                for item in content
                if "text" in item
            ),
            "",
        )

        usage = response.get("usage", {})
        metrics = response.get("metrics", {})

        return ModelResponse(
            text=text,
            model_id=self.model_id,
            input_tokens=usage.get("inputTokens", 0),
            output_tokens=usage.get("outputTokens", 0),
            total_tokens=usage.get("totalTokens", 0),
            latency_ms=metrics.get("latencyMs", 0),
        )
    