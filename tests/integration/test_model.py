"""
Smoke-test Amazon Bedrock model invocation with brief prompts.

This script sends a few short test prompts to Amazon Nova Pro through the
reusable ModelInvoker and prints the response text and inference metrics
for each one.

It does not read from S3 or require Terraform, so it is a quick and cheap
way to confirm that credentials, the model ID, and the Converse API call
work before running the full claim-extraction flow.
"""

from __future__ import annotations

from claim_poc.config import config
from claim_poc.model_invoker import ModelInvoker


TEST_PROMPTS = [
    "Hello",

    "Extract the claim number.",

    """
Extract the following fields from an insurance claim:

- claim_number
- policy_number
- claim_amount

Return JSON only.
""".strip(),
]


def main() -> None:
    """Send each brief test prompt to the model and print the results."""

    model = ModelInvoker()

    print(f"Model:   {model.model_id}")
    print(f"Region:  {config.aws_region}")
    print(f"Profile: {config.aws_profile}")

    total_input_tokens = 0
    total_output_tokens = 0
    total_tokens = 0

    for index, test_prompt in enumerate(TEST_PROMPTS, start=1):
        response = model.invoke(
            test_prompt,
            temperature=0.0,
            max_tokens=100,
        )

        total_input_tokens += response.input_tokens
        total_output_tokens += response.output_tokens
        total_tokens += response.total_tokens

        print(f"\n--- Test {index} ---")

        print("\nPROMPT:")
        print(test_prompt)

        print("\nRESPONSE:")
        print(response.text)

        print("\nMETRICS:")
        print(f"Input tokens:  {response.input_tokens}")
        print(f"Output tokens: {response.output_tokens}")
        print(f"Total tokens:  {response.total_tokens}")
        print(f"Latency:       {response.latency_ms} ms")

    print("\n--- Summary ---")
    print(f"Prompts run:         {len(TEST_PROMPTS)}")
    print(f"Total input tokens:  {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total tokens:        {total_tokens}")


if __name__ == "__main__":
    main()