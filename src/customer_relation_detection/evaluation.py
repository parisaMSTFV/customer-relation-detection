"""Threshold selection and pair-level entity-resolution evaluation."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from customer_relation_detection.clustering import guarded_component_assignments


def binary_metrics(actual: pd.Series, predicted: pd.Series) -> dict[str, float | int]:
    """Return confusion counts and positive-class precision, recall, and F1."""
    actual_values = actual.astype(int)
    predicted_values = predicted.astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        actual_values,
        predicted_values,
        average="binary",
        zero_division=0,
    )
    tp = int(((actual_values == 1) & (predicted_values == 1)).sum())
    fp = int(((actual_values == 0) & (predicted_values == 1)).sum())
    fn = int(((actual_values == 1) & (predicted_values == 0)).sum())
    tn = int(((actual_values == 0) & (predicted_values == 0)).sum())
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


def select_threshold(
    validation_pairs: pd.DataFrame,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
) -> tuple[float, pd.DataFrame]:
    """Choose the highest-precision threshold among validation F1 ties."""
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        metrics = binary_metrics(
            validation_pairs["is_related"],
            validation_pairs["relation_score"].ge(threshold),
        )
        rows.append({"threshold": float(round(threshold, 10)), **metrics})
    curve = pd.DataFrame.from_records(rows)
    best_f1 = curve["f1"].max()
    candidates = curve[np.isclose(curve["f1"], best_f1)]
    best_precision = candidates["precision"].max()
    selected = candidates[np.isclose(candidates["precision"], best_precision)].sort_values(
        "threshold", ascending=False
    )
    return float(selected.iloc[0]["threshold"]), curve


def _co_membership_pairs(frame: pd.DataFrame, group_column: str) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for _, group in frame.groupby(group_column):
        pairs.update(combinations(sorted(group["account_id"].astype(str)), 2))
    return pairs


def component_pair_metrics(
    assignments: pd.DataFrame,
    truth: pd.DataFrame,
) -> dict[str, float | int]:
    """Measure component false merges and false splits over all account pairs."""
    merged = truth[["account_id", "true_group_id"]].merge(
        assignments, on="account_id", validate="one_to_one"
    )
    if len(merged) != len(truth):
        raise ValueError("Assignments must cover every truth account")
    actual = _co_membership_pairs(merged, "true_group_id")
    predicted = _co_membership_pairs(merged, "predicted_group_id")
    true_positives = len(actual & predicted)
    false_positives = len(predicted - actual)
    false_negatives = len(actual - predicted)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_merges": false_positives,
        "false_splits": false_negatives,
    }


def bcubed_metrics(assignments: pd.DataFrame, truth: pd.DataFrame) -> dict[str, float]:
    """Return account-weighted B-cubed precision, recall, and F1."""
    merged = truth[["account_id", "true_group_id"]].merge(
        assignments, on="account_id", validate="one_to_one"
    )
    if len(merged) != len(truth):
        raise ValueError("Assignments must cover every truth account")
    intersections = merged.groupby(["predicted_group_id", "true_group_id"])["account_id"].transform(
        "size"
    )
    predicted_sizes = merged.groupby("predicted_group_id")["account_id"].transform("size")
    true_sizes = merged.groupby("true_group_id")["account_id"].transform("size")
    precision = float((intersections / predicted_sizes).mean())
    recall = float((intersections / true_sizes).mean())
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def select_component_policy(
    validation_pairs: pd.DataFrame,
    validation_truth: pd.DataFrame,
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_step: float,
    component_sizes: tuple[int, ...],
    precision_floor: float,
    false_merge_cost: float,
    false_split_cost: float,
) -> tuple[float, int, pd.DataFrame]:
    """Select threshold and cap using graph outcomes and asymmetric error costs."""
    thresholds = np.arange(threshold_min, threshold_max + threshold_step / 2, threshold_step)
    accounts = sorted(validation_truth["account_id"].astype(str))
    rows: list[dict[str, float | int | bool]] = []
    for threshold in thresholds:
        scored = validation_pairs.copy()
        scored["enhanced_match"] = scored["relation_score"].ge(threshold).astype(int)
        for component_size in component_sizes:
            assignments, audit = guarded_component_assignments(
                accounts,
                scored,
                prediction_column="enhanced_match",
                score_column="relation_score",
                max_component_size=component_size,
            )
            pair_metrics = component_pair_metrics(assignments, validation_truth)
            bcubed = bcubed_metrics(assignments, validation_truth)
            cost = false_merge_cost * int(pair_metrics["false_merges"]) + false_split_cost * int(
                pair_metrics["false_splits"]
            )
            rows.append(
                {
                    "threshold": float(round(threshold, 10)),
                    "max_component_size": int(component_size),
                    "component_precision": float(pair_metrics["precision"]),
                    "component_recall": float(pair_metrics["recall"]),
                    "component_f1": float(pair_metrics["f1"]),
                    "false_merges": int(pair_metrics["false_merges"]),
                    "false_splits": int(pair_metrics["false_splits"]),
                    "bcubed_precision": bcubed["precision"],
                    "bcubed_recall": bcubed["recall"],
                    "bcubed_f1": bcubed["f1"],
                    "weighted_error_cost": float(cost),
                    "deferred_edges": int(
                        audit["guardrail_decision"].eq("deferred_component_cap").sum()
                    ),
                    "meets_precision_floor": bool(pair_metrics["precision"] >= precision_floor),
                }
            )
    curve = pd.DataFrame.from_records(rows)
    eligible = curve[curve["meets_precision_floor"]]
    if eligible.empty:
        raise ValueError("No component policy meets the configured precision floor")
    selected = eligible.sort_values(
        ["weighted_error_cost", "component_f1", "threshold", "max_component_size"],
        ascending=[True, False, False, True],
        kind="mergesort",
    ).iloc[0]
    curve["selected"] = np.isclose(curve["threshold"], float(selected["threshold"])) & curve[
        "max_component_size"
    ].eq(int(selected["max_component_size"]))
    return float(selected["threshold"]), int(selected["max_component_size"]), curve


def evaluate_pair_models(
    test_pairs: pd.DataFrame,
    threshold: float,
) -> tuple[pd.DataFrame, dict[str, dict[str, float | int]]]:
    """Compare exact-key baseline and scored relation pairs on held-out buildings."""
    result = test_pairs.copy()
    result["enhanced_match"] = result["relation_score"].ge(threshold).astype(int)
    summary = {
        "exact_key_baseline": binary_metrics(result["is_related"], result["baseline_match"]),
        "scored_resolution": binary_metrics(result["is_related"], result["enhanced_match"]),
    }
    return result, summary


def recall_by_noise(test_pairs: pd.DataFrame) -> pd.DataFrame:
    """Calculate positive-pair recall for each address-noise severity."""
    positives = test_pairs[test_pairs["is_related"] == 1]
    rows: list[dict[str, object]] = []
    for noise_level, group in positives.groupby("pair_noise_level"):
        rows.append(
            {
                "noise_level": noise_level,
                "positive_pairs": len(group),
                "baseline_recall": float(group["baseline_match"].mean()),
                "enhanced_recall": float(group["enhanced_match"].mean()),
            }
        )
    order = pd.CategoricalDtype(["clean", "moderate", "heavy"], ordered=True)
    result = pd.DataFrame.from_records(rows)
    result["noise_level"] = result["noise_level"].astype(order)
    return result.sort_values("noise_level", ignore_index=True)


def error_metrics_by_noise(test_pairs: pd.DataFrame) -> pd.DataFrame:
    """Report precision, recall, and F1 for each synthetic noise subgroup."""
    rows: list[dict[str, object]] = []
    for noise_level, group in test_pairs.groupby("pair_noise_level"):
        for model, prediction in (
            ("exact_key_baseline", "baseline_match"),
            ("scored_resolution", "enhanced_match"),
            ("guarded_resolution", "guardrail_match"),
        ):
            if prediction not in group:
                continue
            metrics = binary_metrics(group["is_related"], group[prediction])
            rows.append(
                {
                    "noise_level": noise_level,
                    "model": model,
                    "pairs": len(group),
                    **metrics,
                }
            )
    order = pd.CategoricalDtype(["clean", "moderate", "heavy"], ordered=True)
    result = pd.DataFrame.from_records(rows)
    result["noise_level"] = result["noise_level"].astype(order)
    return result.sort_values(["noise_level", "model"], ignore_index=True)
