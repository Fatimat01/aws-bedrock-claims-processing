"""
Unit tests for the claim processor response helpers.

These tests verify that model responses are parsed into JSON objects, that
markdown code fences are tolerated, and that generated summaries are
normalized into a single clean line of text.

The tests do not call AWS services and can run entirely locally.
"""

from __future__ import annotations

import json

import pytest

from claim_poc.processor import (
    normalize_summary,
    parse_json_response,
    strip_code_fence,
)


def test_plain_json_is_parsed() -> None:
    """Verify that an unwrapped JSON object is parsed."""

    assert parse_json_response('{"claim_number": "CLM-1"}') == {
        "claim_number": "CLM-1"
    }


def test_fenced_json_is_parsed() -> None:
    """Verify that JSON wrapped in markdown fences is parsed."""

    fenced = '```json\n{"claim_number": "CLM-1"}\n```'

    assert parse_json_response(fenced) == {"claim_number": "CLM-1"}


def test_unlabelled_fence_is_stripped() -> None:
    """Verify that a fence without a language label is stripped."""

    assert strip_code_fence('```\n{"a": 1}\n```') == '{"a": 1}'


def test_unfenced_text_is_unchanged() -> None:
    """Verify that text without a fence is returned as-is."""

    assert strip_code_fence('{"a": 1}') == '{"a": 1}'


def test_invalid_json_is_rejected() -> None:
    """Verify that a non-JSON response raises a clear error."""

    with pytest.raises(ValueError):
        parse_json_response("Sorry, I cannot help with that.")


def test_json_array_is_rejected() -> None:
    """Verify that a JSON array is rejected because an object is required."""

    with pytest.raises(ValueError):
        parse_json_response("[1, 2, 3]")


def test_summary_newlines_are_collapsed() -> None:
    """Verify that multi-line summaries become a single clean line."""

    raw = (
        "Claim Summary:\n\n"
        "Claim Type: Auto Collision\n"
        "Claim Amount: $12,500\n"
    )

    summary = normalize_summary(raw)

    assert "\n" not in summary
    assert "  " not in summary
    assert summary == (
        "Claim Summary: Claim Type: Auto Collision "
        "Claim Amount: $12,500"
    )


def test_normalized_summary_has_no_escaped_newlines_in_json() -> None:
    """Verify that the persisted JSON carries no escaped newlines."""

    summary = normalize_summary("Line one.\nLine two.\r\nLine three.")

    assert "\\n" not in json.dumps({"summary": summary})


def test_summary_whitespace_is_trimmed() -> None:
    """Verify that surrounding and repeated whitespace is removed."""

    assert normalize_summary("   padded\t\tsummary   ") == "padded summary"
