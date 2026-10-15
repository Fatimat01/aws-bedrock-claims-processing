"""
Prompt template management for the claim processing application.

This module stores the prompt templates used by the application and renders
them with runtime values. Keeping prompts in one place makes it easy to
compare wording across models and to version prompt changes independently
of the invocation and storage logic.
"""

from __future__ import annotations


EXTRACT_CLAIM_TEMPLATE = """
You are an insurance claims processing assistant.

Extract the structured fields below from the claim document and return
them as a single JSON object.

Rules:
- Return only the JSON object. Do not include markdown code fences,
  explanations, or any text before or after the JSON.
- Use exactly these keys: claim_number, policy_number, claim_date,
  claim_type, claim_amount, incident_severity, claimant_age,
  claimant_name.
- claim_date must be in YYYY-MM-DD format as written in the document.
- claim_amount must be a number with no currency symbol or thousands
  separators.
- claimant_age must be an integer.
- If a value is missing or stated as "Not provided", use null.
- Do not guess or invent values that are not in the document.

Claim document:
<document>
{document_text}
</document>
""".strip()


GENERATE_SUMMARY_TEMPLATE = """
You are an insurance claims processing assistant.

Write a concise summary of the claim below for a claims adjuster, grounded
strictly in the policy document provided.

Rules:

1. Do not approve, deny, or determine coverage for the claim.
2. State only the coverage, limits, deductibles, requirements, and
   exclusions explicitly present in the policy.
3. Do not determine whether an exclusion applies.
4. Do not infer facts that are not provided.
5. If relevant information is missing, state that clearly.
6. Keep the summary concise and factual.

Output format:

- Respond with plain text only. Do not use markdown, headings, bullet
  points, or numbered lists.
- Write the summary as a single paragraph with no line breaks.

Claim data:
<claim>
{claim_data}
</claim>

Policy document:
<policy>
{policy_context}
</policy>
""".strip()


class PromptTemplateManager:
    """Store and render the prompt templates used by the application."""

    def __init__(
        self,
        templates: dict[str, str] | None = None,
    ) -> None:
        self.templates: dict[str, str] = {
            "extract_claim": EXTRACT_CLAIM_TEMPLATE,
            "generate_summary": GENERATE_SUMMARY_TEMPLATE,
        }

        if templates:
            self.templates.update(templates)

    def list_templates(self) -> list[str]:
        """Return the names of all registered templates."""

        return sorted(self.templates)

    def register(
        self,
        name: str,
        template: str,
    ) -> None:
        """
        Add or replace a prompt template.

        Args:
            name: Template name used with get_prompt.
            template: Template text using {placeholder} variables.
        """

        self.templates[name] = template

    def get_prompt(
        self,
        name: str,
        **variables: object,
    ) -> str:
        """
        Render a registered template with the given variables.

        Args:
            name: Name of the registered template.
            **variables: Values substituted into the template placeholders.

        Returns:
            The rendered prompt text.

        Raises:
            KeyError: If no template is registered under the given name.
            ValueError: If the template requires a variable that was not
                provided.
        """

        try:
            template = self.templates[name]

        except KeyError as error:
            raise KeyError(
                f"Unknown prompt template: {name!r}. "
                f"Available templates: {self.list_templates()}"
            ) from error

        try:
            return template.format(**variables)

        except KeyError as error:
            missing = error.args[0]

            raise ValueError(
                f"Prompt template {name!r} requires "
                f"variable {missing!r}, which was not provided."
            ) from error
