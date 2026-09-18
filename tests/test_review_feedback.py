import json

import pandas as pd
import pytest

from customer_relation_detection.review_feedback import (
    ReviewFeedbackError,
    review_feedback_metrics,
    run_review_feedback_audit,
)


def _ref(value: int) -> str:
    return f"REF-{value:020x}"


def _guardrail() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "account_ref_a": _ref(1),
                "account_ref_b": _ref(2),
                "guardrail_decision": "accepted_merge",
            },
            {
                "account_ref_a": _ref(2),
                "account_ref_b": _ref(3),
                "guardrail_decision": "accepted_redundant",
            },
            {
                "account_ref_a": _ref(3),
                "account_ref_b": _ref(4),
                "guardrail_decision": "deferred_component_cap",
            },
        ]
    )


def test_review_feedback_reports_observed_accepted_edge_quality() -> None:
    feedback = pd.DataFrame(
        [
            {
                "account_ref_a": _ref(2),
                "account_ref_b": _ref(1),
                "review_outcome": "match",
            },
            {
                "account_ref_a": _ref(2),
                "account_ref_b": _ref(3),
                "review_outcome": "not_match",
            },
            {
                "account_ref_a": _ref(3),
                "account_ref_b": _ref(4),
                "review_outcome": "uncertain",
            },
        ]
    )
    metrics = review_feedback_metrics(_guardrail(), feedback)
    assert metrics["overall"]["review_coverage"] == 1.0
    assert metrics["accepted_edges"]["observed_precision"] == 0.5
    assert metrics["accepted_edges"]["observed_false_merge_rate"] == 0.5
    assert metrics["deferred_edges"]["decisive_edges"] == 0
    assert metrics["deferred_edges"]["observed_match_rate"] is None


def test_review_feedback_rejects_unknown_pairs() -> None:
    feedback = pd.DataFrame(
        [
            {
                "account_ref_a": _ref(8),
                "account_ref_b": _ref(9),
                "review_outcome": "match",
            }
        ]
    )
    with pytest.raises(ReviewFeedbackError, match="not present"):
        review_feedback_metrics(_guardrail(), feedback)


def test_review_feedback_rejects_duplicate_pairs_after_canonicalization() -> None:
    feedback = pd.DataFrame(
        [
            {
                "account_ref_a": _ref(1),
                "account_ref_b": _ref(2),
                "review_outcome": "match",
            },
            {
                "account_ref_a": _ref(2),
                "account_ref_b": _ref(1),
                "review_outcome": "not_match",
            },
        ]
    )
    with pytest.raises(ReviewFeedbackError, match="only once"):
        review_feedback_metrics(_guardrail(), feedback)


def test_review_feedback_audit_exports_only_aggregate_json(tmp_path) -> None:
    feedback = pd.DataFrame(
        [
            {
                "account_ref_a": _ref(1),
                "account_ref_b": _ref(2),
                "review_outcome": "match",
            }
        ]
    )
    run_review_feedback_audit(_guardrail(), feedback, tmp_path)
    output = tmp_path / "reports" / "review_feedback_metrics.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    raw = output.read_text(encoding="utf-8")
    assert payload["accepted_edges"]["observed_precision"] == 1.0
    assert "REF-" not in raw
