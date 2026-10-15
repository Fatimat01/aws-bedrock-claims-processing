"""
Compare Amazon Bedrock models on the claim processing task.

Each model runs the full pipeline over every sample claim. Extraction is
scored field by field against the ground truth in S3, and the generated
summary is checked against the guardrail rules in the summary prompt.

Token counts are reported instead of dollar cost, because Bedrock rates
vary by model and region.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from claim_poc.config import config, get_bucket_name
from claim_poc.model_invoker import ModelInvoker
from claim_poc.policy_retriever import PolicyRetriever
from claim_poc.processor import normalize_summary, parse_json_response
from claim_poc.prompt_manager import PromptTemplateManager
from claim_poc.storage import S3Storage
from claim_poc.validator import ClaimValidator


# Candidates across cost tiers and vendors. temperature=None omits the
# sampling parameter, which newer reasoning models reject.
MODELS = [
    ("Nova Micro", "us.amazon.nova-micro-v1:0", 0.0),
    ("Nova Lite", "us.amazon.nova-lite-v1:0", 0.0),
    ("Nova Pro", "us.amazon.nova-pro-v1:0", 0.0),
    ("Nova 2 Lite", "us.amazon.nova-2-lite-v1:0", 0.0),
    ("Claude Haiku 4.5", "us.anthropic.claude-haiku-4-5-20251001-v1:0", 0.0),
    ("Claude Sonnet 5", "us.anthropic.claude-sonnet-5", None),
    ("Llama 4 Scout", "us.meta.llama4-scout-17b-instruct-v1:0", 0.0),
]

SCORED_FIELDS = [
    "claim_number",
    "policy_number",
    "claim_date",
    "claim_type",
    "claim_amount",
    "incident_severity",
    "claimant_age",
    "claimant_name",
]

# Summaries must not decide coverage. These patterns are subject-aware on
# purpose: quoting the policy ("fire damage is covered subject to policy
# terms") is required by the prompt, so a bare "is covered" match produced
# false positives. Each pattern needs the claim itself as the subject.
GUARDRAILS = {
    "coverage_determination": r"\b(this\s+|the\s+)?claim\b[^.]{0,70}?\b"
    r"(is|was|are|appears\s+to\s+be|falls\s+under)\s+"
    r"(fully\s+|partially\s+|likely\s+|not\s+)?(covered|this\s+coverage)\b",
    "approval_language": r"\b(approve[sd]?|approval|den(y|ies|ied|ial)"
    r"|reject(ed)?|payable|should\s+be\s+paid)\b",
    "exclusion_determination": r"\b(no\s+)?exclusions?\s+(appl(y|ies|ied)"
    r"|do\s+not\s+appl|would\s+appl|are\s+relevant)",
    "recommendation": r"\brecommend(s|ed|ation)?\b[^.]{0,40}?\b"
    r"(claim|payment|approv|settle|adjuster\s+should)",
    "markdown_formatting": r"(^|\n)\s*(#{1,6}\s|\*\s|-\s|\d+\.\s)|\*\*",
}


@dataclass
class ClaimResult:
    """
    Outcome of running one claim document through one model.

    Holds everything measured for a single claim so the per-model totals can
    be derived by summing these. The defaults describe a run that failed
    before producing anything, so a record is always complete enough to
    aggregate.

    Attributes:
        fields_correct: Extracted fields matching ground truth, out of
            eight. Scored even when validation later fails.
        validated: Whether the extracted claim passed the Pydantic schema.
        guardrails: Rules the generated summary broke, each with evidence.
        input_tokens: Prompt tokens across both inference calls.
        output_tokens: Generated tokens across both inference calls.
        latency_ms: Combined Bedrock latency for both calls.
        summary: The normalized summary text, kept for manual review.
        error: Exception type and message if any stage failed, else None.
    """

    fields_correct: int = 0
    validated: bool = False
    guardrails: list[dict[str, str]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    summary: str = ""
    error: str | None = None


def values_match(actual: Any, expected: Any) -> bool:
    """
    Compare one extracted field against its ground-truth value.

    Comparison is type-tolerant because models return the same value in
    different shapes. An amount may arrive as 12500, 12500.0, or "12500",
    and all three should count as correct. Numbers are therefore compared
    numerically and everything else as trimmed strings.

    Args:
        actual: Value the model extracted for this field.
        expected: Ground-truth value from the expected JSON in S3.

    Returns:
        True if the two represent the same value. Two nulls match each
        other, but a null never matches a present value.
    """

    if actual is None or expected is None:
        return actual is None and expected is None

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        try:
            return float(actual) == float(expected)
        except (TypeError, ValueError):
            return False

    return str(actual).strip() == str(expected).strip()


def check_guardrails(summary: str, policy: str) -> list[dict[str, str]]:
    """
    Find places where a generated summary breaks the prompt's rules.

    The summary prompt forbids deciding coverage, approving or denying,
    ruling on exclusions, recommending an outcome, and using markdown. This
    scans for each of those and returns the sentence that triggered it, so a
    reviewer can judge the match rather than trust a bare count.

    Quoting the policy is explicitly required by the prompt, so any sentence
    reproduced from the policy document is skipped. Without that check, a
    model writing "fire damage is covered subject to policy terms" would be
    flagged for doing exactly what it was asked to do.

    Args:
        summary: Normalized summary text produced by the model.
        policy: Policy document the summary was grounded in, used to tell
            permitted quotation apart from the model's own determination.

    Returns:
        One entry per rule broken, each with the rule name and the offending
        sentence. Empty when the summary is compliant.
    """

    found = []

    for rule, pattern in GUARDRAILS.items():
        match = re.search(pattern, summary, re.IGNORECASE)

        if not match:
            continue

        start = summary.rfind(".", 0, match.start()) + 1
        end = summary.find(".", match.end())
        sentence = summary[start : end if end != -1 else len(summary)].strip()

        if sentence and sentence.lower() in policy.lower():
            continue

        found.append({"rule": rule, "sentence": sentence})

    return found


def run_claim(
    model: ModelInvoker,
    temperature: float | None,
    document: str,
    expected: dict[str, Any],
    prompts: PromptTemplateManager,
    validator: ClaimValidator,
    retriever: PolicyRetriever,
) -> ClaimResult:
    """
    Run one claim through the full pipeline and score what comes back.

    This is the unit of measurement for the whole comparison. It performs
    the same six stages as the production processor, extraction, parsing,
    scoring, validation, policy retrieval, and summary generation, while
    recording tokens and latency along the way.

    Scoring happens before validation on purpose. A model can extract every
    field correctly and still fail the schema on a technicality, and the
    accuracy number should reflect what it actually read off the document.

    Any failure stops this claim rather than raising, so one bad model never
    aborts the run. The recorded exception type identifies the stage that
    broke.

    Args:
        model: Invoker already bound to the model under test.
        temperature: Sampling temperature, or None for models that reject
            sampling parameters.
        document: Raw claim document text.
        expected: Ground-truth field values for this claim.
        prompts: Template manager supplying both prompts.
        validator: Schema validator applied to the extracted claim.
        retriever: Policy lookup used to ground the summary.

    Returns:
        A ClaimResult holding the score, token counts, latency, the summary
        text, and an error message if any stage failed.
    """

    result = ClaimResult()

    def call(prompt: str, max_tokens: int):
        response = model.invoke(
            prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        result.input_tokens += response.input_tokens
        result.output_tokens += response.output_tokens
        result.latency_ms += response.latency_ms
        return response

    # The exception type identifies the stage that failed: RuntimeError from
    # Bedrock, ValueError from JSON parsing, ValidationError from the schema,
    # FileNotFoundError from policy lookup.
    try:
        extraction = call(
            prompts.get_prompt("extract_claim", document_text=document),
            1000,
        )

        claim = parse_json_response(extraction.text)

        result.fields_correct = sum(
            values_match(claim.get(f), expected.get(f))
            for f in SCORED_FIELDS
        )

        validated = validator.validate(claim)
        result.validated = True

        policy = retriever.retrieve(validated.policy_number)

        summary = call(
            prompts.get_prompt(
                "generate_summary",
                claim_data=json.dumps(
                    validated.model_dump(mode="json"),
                    indent=2,
                ),
                policy_context=policy,
            ),
            500,
        )

        result.summary = normalize_summary(summary.text)
        result.guardrails = check_guardrails(result.summary, policy)

    except (
        RuntimeError,
        ValueError,
        ValidationError,
        FileNotFoundError,
    ) as error:
        result.error = f"{type(error).__name__}: {error}"

    return result


def run_model(
    label: str,
    model_id: str,
    temperature: float | None,
    documents: dict[str, str],
    expectations: dict[str, dict[str, Any]],
    storage: S3Storage,
    repeats: int,
) -> dict[str, Any]:
    """
    Evaluate one model across every claim and aggregate the results.

    Repeating the claim set is what separates a stable result from a lucky
    one. Models are not fully deterministic even at temperature zero, so a
    single pass cannot show run-to-run drift.

    Progress is printed as each claim finishes, because a full run makes
    dozens of Bedrock calls and silence would be hard to distinguish from a
    stall.

    If the account cannot invoke the model, this returns early rather than
    repeating the same access error for every claim. Bedrock lists models an
    account is not entitled to use, so this is expected rather than
    exceptional.

    Args:
        label: Human-readable model name used in output.
        model_id: Bedrock model or inference profile identifier.
        temperature: Sampling temperature, or None to omit it.
        documents: Claim text keyed by claim ID.
        expectations: Ground-truth values keyed by claim ID.
        storage: S3 client used to retrieve policy documents.
        repeats: Times to run the whole claim set.

    Returns:
        A dictionary carrying the model identity, an availability flag, each
        individual run, and the aggregated totals. When the model is
        unavailable, only the identity, the flag, and the error are present.
    """

    print(f"\n--- {label} ({model_id}) ---", flush=True)

    prompts = PromptTemplateManager()
    validator = ClaimValidator()
    retriever = PolicyRetriever(storage)
    model = ModelInvoker(model_id=model_id)

    runs: dict[str, ClaimResult] = {}

    for attempt in range(1, repeats + 1):
        for claim_id, document in documents.items():
            result = run_claim(
                model=model,
                temperature=temperature,
                document=document,
                expected=expectations[claim_id],
                prompts=prompts,
                validator=validator,
                retriever=retriever,
            )

            # The account may list a model it is not entitled to invoke.
            # Report it once instead of repeating the same error per claim.
            if result.error and "AccessDenied" in result.error:
                print("  not available to this account, skipping", flush=True)
                return {
                    "label": label,
                    "model_id": model_id,
                    "available": False,
                    "error": result.error,
                    "runs": {},
                }

            key = claim_id if repeats == 1 else f"{claim_id}#{attempt}"
            runs[key] = result

            print(
                f"  {key}: {'FAIL' if result.error else 'ok'} "
                f"fields {result.fields_correct}/{len(SCORED_FIELDS)} "
                f"guardrails {len(result.guardrails)} "
                f"{result.latency_ms} ms",
                flush=True,
            )

            if result.error:
                print(f"    {result.error}", flush=True)

    values = list(runs.values())
    latencies = [r.latency_ms for r in values]
    total_fields = len(values) * len(SCORED_FIELDS)
    correct = sum(r.fields_correct for r in values)

    return {
        "label": label,
        "model_id": model_id,
        "available": True,
        "runs": {k: asdict(v) for k, v in runs.items()},
        "totals": {
            "runs": len(values),
            "fields_correct": correct,
            "fields_total": total_fields,
            "accuracy": round(correct / total_fields, 4),
            "validated": sum(r.validated for r in values),
            "guardrails": sum(len(r.guardrails) for r in values),
            "input_tokens": sum(r.input_tokens for r in values),
            "output_tokens": sum(r.output_tokens for r in values),
            "mean_latency_ms": round(sum(latencies) / len(latencies)),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
        },
    }


def print_table(results: list[dict[str, Any]]) -> None:
    """
    Print a side-by-side comparison of every model that ran.

    Rows are ordered by accuracy, then by mean latency, so the fastest of
    several equally accurate models comes first. That ordering matters here
    because accuracy tends to saturate on clean documents, leaving speed and
    token cost as the real differentiators.

    Models the account could not invoke are listed separately rather than
    shown as zero-accuracy rows, which would otherwise distort the ranking.

    Args:
        results: Per-model dictionaries as returned by run_model.
    """

    evaluated = [r for r in results if r["available"]]
    skipped = [r for r in results if not r["available"]]

    print("\n" + "=" * 92)
    print(
        f"{'Model':<20}{'Fields':>9}{'Acc':>8}{'Valid':>7}{'Guard':>7}"
        f"{'In tok':>9}{'Out tok':>9}{'Latency':>11}{'Range':>13}"
    )
    print("-" * 92)

    ranked = sorted(
        evaluated,
        key=lambda r: (-r["totals"]["accuracy"], r["totals"]["mean_latency_ms"]),
    )

    for result in ranked:
        t = result["totals"]
        print(
            f"{result['label']:<20}"
            f"{t['fields_correct']:>4}/{t['fields_total']:<4}"
            f"{t['accuracy'] * 100:>7.1f}%"
            f"{t['validated']:>4}/{t['runs']:<2}"
            f"{t['guardrails']:>7}"
            f"{t['input_tokens']:>9}"
            f"{t['output_tokens']:>9}"
            f"{t['mean_latency_ms']:>8} ms"
            f"{t['min_latency_ms']:>7}-{t['max_latency_ms']}"
        )

    if skipped:
        print("\nSkipped, not available to this account:")
        for result in skipped:
            print(f"  {result['label']} ({result['model_id']})")

    print(
        "\nFields = correct/total, Valid = runs passing schema validation, "
        "Guard = guardrail flags."
    )


def main() -> int:
    """
    Load the sample data, evaluate every candidate model, and save the run.

    Running with no arguments is the intended path. It evaluates the full
    candidate list against the configured sample claims, repeated enough
    times to expose drift. The flags exist to narrow a run while iterating,
    since a full pass makes dozens of billable Bedrock calls.

    The complete run, including every individual claim result, is written to
    the evaluations prefix in S3 so a comparison can be re-read later rather
    than re-run.

    Returns:
        Zero on success. One if no model matched the filter, or if the
        sample data has not been generated yet.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claim-ids", nargs="+",
                        default=list(config.selected_claim_ids))
    parser.add_argument("--models", nargs="+",
                        help="Model IDs to evaluate. Defaults to all.")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Times to repeat the claim set per model "
                             "(default 3, enough to show run-to-run drift).")
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip persisting the run to S3.")
    args = parser.parse_args()

    candidates = MODELS

    if args.models:
        wanted = set(args.models)
        candidates = [m for m in MODELS if m[1] in wanted]

        if not candidates:
            print("No models matched --models.", file=sys.stderr)
            return 1

    bucket = get_bucket_name()
    storage = S3Storage(bucket=bucket)

    print(f"Bucket:  {bucket}")
    print(f"Profile: {config.aws_profile}  Region: {config.aws_region}")
    print(f"Claims:  {', '.join(args.claim_ids)}")
    print(f"Models:  {len(candidates)}  Repeats: {args.repeats}")

    try:
        documents = {
            claim_id: storage.read_text(config.claim_document_key(claim_id))
            for claim_id in args.claim_ids
        }
        expectations = {
            claim_id: json.loads(
                storage.read_text(config.claim_expected_key(claim_id))
            )
            for claim_id in args.claim_ids
        }
    except FileNotFoundError as error:
        print(
            f"\nSample data missing: {error}\n"
            "Run scripts/build-claims.py and scripts/build-policies.py "
            "first.",
            file=sys.stderr,
        )
        return 1

    results = [
        run_model(
            label=label,
            model_id=model_id,
            temperature=temperature,
            documents=documents,
            expectations=expectations,
            storage=storage,
            repeats=args.repeats,
        )
        for label, model_id, temperature in candidates
    ]

    print_table(results)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    if not args.no_upload:
        uri = storage.upload_json(
            data={
                "run_id": run_id,
                "claim_ids": args.claim_ids,
                "repeats": args.repeats,
                "scored_fields": SCORED_FIELDS,
                "results": results,
            },
            key=config.evaluation_key(run_id),
        )
        print(f"\nSaved to {uri}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
