"""Aggregate quality audits for pseudonymized human review feedback."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

REFERENCE_PATTERN = re.compile(r"^REF-[0-9a-f]{20}$")
FEEDBACK_COLUMNS = {"account_ref_a", "account_ref_b", "review_outcome"}
GUARDRAIL_COLUMNS = {"account_ref_a", "account_ref_b", "guardrail_decision"}
SUPPORTED_OUTCOMES = {"match", "not_match", "uncertain"}
SUPPORTED_DECISIONS = {
    "accepted_merge",
    "accepted_redundant",
    "deferred_component_cap",
}


class ReviewFeedbackError(ValueError):
    """Raised when review feedback violates the pseudonymized audit contract."""


def _canonicalize_pairs(
    frame: pd.DataFrame,
    *,
    required_columns: set[str],
    outcome_column: str | None = None,
) -> pd.DataFrame:
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ReviewFeedbackError(f"Missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ReviewFeedbackError("Input must not be empty")

    result = frame.copy()
    for column in ("account_ref_a", "account_ref_b"):
        result[column] = result[column].astype("string").str.strip()
        if result[column].isna().any() or result[column].eq("").any():
            raise ReviewFeedbackError(f"{column} must not be null or blank")
        if not result[column].map(lambda value: bool(REFERENCE_PATTERN.fullmatch(str(value)))).all():
            raise ReviewFeedbackError(
                f"{column} must contain pseudonymous references in REF-<20 hex> format"
            )

    if result["account_ref_a"].eq(result["account_ref_b"]).any():
        raise ReviewFeedbackError("Self-pairs are not allowed")

    canonical = [
        tuple(sorted((str(left), str(right))))
        for left, right in zip(
            result["account_ref_a"],
            result["account_ref_b"],
            strict=True,
        )
    ]
    result["pair_left"] = [pair[0] for pair in canonical]
    result["pair_right"] = [pair[1] for pair in canonical]
    if result.duplicated(["pair_left", "pair_right"]).any():
        raise ReviewFeedbackError("Each account pair may appear only once")

    if outcome_column is not None:
        result[outcome_column] = result[outcome_column].astype("string").str.strip().str.lower()
        if result[outcome_column].isna().any() or result[outcome_column].eq("").any():
            raise ReviewFeedbackError(f"{outcome_column} must not be null or blank")
        unsupported = set(result[outcome_column]).difference(SUPPORTED_OUTCOMES)
        if unsupported:
            raise ReviewFeedbackError(
                f"Unsupported review outcomes: {sorted(str(value) for value in unsupported)}"
            )
    return result


def validate_review_inputs(
    guardrail: pd.DataFrame,
    feedback: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate and canonicalize a pseudonymized guardrail export and reviewer feedback."""
    safe_guardrail = _canonicalize_pairs(guardrail, required_columns=GUARDRAIL_COLUMNS)
    safe_feedback = _canonicalize_pairs(
        feedback,
        required_columns=FEEDBACK_COLUMNS,
        outcome_column="review_outcome",
    )

    safe_guardrail["guardrail_decision"] = (
        safe_guardrail["guardrail_decision"].astype("string").str.strip()
    )
    if safe_guardrail["guardrail_decision"].isna().any() or safe_guardrail[
        "guardrail_decision"
    ].eq("").any():
        raise ReviewFeedbackError("guardrail_decision must not be null or blank")
    unsupported_decisions = set(safe_guardrail["guardrail_decision"]).difference(
        SUPPORTED_DECISIONS
    )
    if unsupported_decisions:
        raise ReviewFeedbackError(
            "Unsupported guardrail decisions: "
            f"{sorted(str(value) for value in unsupported_decisions)}"
        )

    guardrail_pairs = set(
        safe_guardrail[["pair_left", "pair_right"]].itertuples(index=False, name=None)
    )
    feedback_pairs = set(
        safe_feedback[["pair_left", "pair_right"]].itertuples(index=False, name=None)
    )
    unknown_pairs = feedback_pairs.difference(guardrail_pairs)
    if unknown_pairs:
        raise ReviewFeedbackError(
            "Review feedback contains pairs that are not present in the guardrail export"
        )
    return safe_guardrail, safe_feedback


def _rate(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def _review_summary(frame: pd.DataFrame) -> dict[str, int | float | None]:
    reviewed = frame["review_outcome"].notna()
    decisive = frame["review_outcome"].isin(["match", "not_match"])
    matches = frame["review_outcome"].eq("match")
    non_matches = frame["review_outcome"].eq("not_match")
    uncertain = frame["review_outcome"].eq("uncertain")
    total = len(frame)
    reviewed_count = int(reviewed.sum())
    decisive_count = int(decisive.sum())
    match_count = int(matches.sum())
    non_match_count = int(non_matches.sum())
    uncertain_count = int(uncertain.sum())
    return {
        "total_edges": total,
        "reviewed_edges": reviewed_count,
        "decisive_edges": decisive_count,
        "confirmed_matches": match_count,
        "confirmed_non_matches": non_match_count,
        "uncertain_edges": uncertain_count,
        "review_coverage": _rate(reviewed_count, total),
        "decisive_share_of_reviewed": _rate(decisive_count, reviewed_count),
        "match_rate_among_decisive": _rate(match_count, decisive_count),
        "non_match_rate_among_decisive": _rate(non_match_count, decisive_count),
    }


def review_feedback_metrics(
    guardrail: pd.DataFrame,
    feedback: pd.DataFrame,
) -> dict[str, Any]:
    """Return aggregate quality metrics without exporting pair references."""
    safe_guardrail, safe_feedback = validate_review_inputs(guardrail, feedback)
    joined = safe_guardrail.merge(
        safe_feedback[["pair_left", "pair_right", "review_outcome"]],
        on=["pair_left", "pair_right"],
        how="left",
        validate="one_to_one",
    )
    accepted = joined[joined["guardrail_decision"].str.startswith("accepted_")]
    deferred = joined[joined["guardrail_decision"].eq("deferred_component_cap")]

    accepted_summary = _review_summary(accepted)
    deferred_summary = _review_summary(deferred)
    by_decision = {
        str(decision): _review_summary(group)
        for decision, group in joined.groupby("guardrail_decision", sort=True)
    }

    digest = hashlib.sha256()
    for frame in (
        safe_guardrail[
            ["pair_left", "pair_right", "guardrail_decision"]
        ].sort_values(["pair_left", "pair_right"], kind="mergesort"),
        safe_feedback[
            ["pair_left", "pair_right", "review_outcome"]
        ].sort_values(["pair_left", "pair_right"], kind="mergesort"),
    ):
        digest.update(frame.to_csv(index=False).encode("utf-8"))

    return {
        "overall": _review_summary(joined),
        "accepted_edges": {
            **accepted_summary,
            "observed_precision": accepted_summary["match_rate_among_decisive"],
            "observed_false_merge_rate": accepted_summary[
                "non_match_rate_among_decisive"
            ],
        },
        "deferred_edges": {
            **deferred_summary,
            "observed_match_rate": deferred_summary["match_rate_among_decisive"],
        },
        "by_guardrail_decision": by_decision,
        "artifact_fingerprint": digest.hexdigest()[:16],
        "evaluation_boundary": (
            "Metrics describe only reviewed high-score guardrail edges. "
            "They do not estimate recall, candidate coverage, full component quality, "
            "fairness, or business impact."
        ),
        "privacy": {
            "raw_account_ids_required": False,
            "raw_addresses_required": False,
            "pair_references_exported": False,
            "aggregate_output_only": True,
        },
    }


def run_review_feedback_audit(
    guardrail: pd.DataFrame,
    feedback: pd.DataFrame,
    output_root: Path,
) -> dict[str, Any]:
    """Write an aggregate review-feedback audit under reports/."""
    metrics = review_feedback_metrics(guardrail, feedback)
    reports_dir = Path(output_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "review_feedback_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return metrics
