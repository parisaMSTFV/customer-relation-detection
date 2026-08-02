"""Threshold selection and pair-level entity-resolution evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


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
