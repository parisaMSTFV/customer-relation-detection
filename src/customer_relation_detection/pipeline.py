"""End-to-end synthetic relation-resolution pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from customer_relation_detection.clustering import (
    build_group_profiles,
    cluster_ari,
    component_assignments,
    guarded_component_assignments,
)
from customer_relation_detection.config import PROJECT_ROOT, AnalysisConfig, load_config
from customer_relation_detection.evaluation import (
    binary_metrics,
    evaluate_pair_models,
    recall_by_noise,
    select_threshold,
)
from customer_relation_detection.normalization import build_account_signatures
from customer_relation_detection.pairs import (
    all_true_related_pairs,
    build_pair_features,
    generate_candidate_pairs,
)
from customer_relation_detection.reporting import (
    plot_model_comparison,
    plot_noise_recall,
    plot_review_graph,
    plot_threshold_curve,
)
from customer_relation_detection.schema import validate_inputs
from customer_relation_detection.synthetic import generate_orders


def _fingerprint(frames: list[pd.DataFrame]) -> str:
    digest = hashlib.sha256()
    for frame in frames:
        digest.update(frame.to_csv(index=False, float_format="%.12g").encode("utf-8"))
    return digest.hexdigest()[:16]


def _mark_guarded_pairs(pairs: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Mark only accepted high-score edges as operational component links."""
    result = pairs.copy()
    accepted = audit[audit["guardrail_decision"].str.startswith("accepted")][
        ["account_a", "account_b"]
    ].copy()
    accepted["guardrail_match"] = 1
    result = result.merge(accepted, on=["account_a", "account_b"], how="left")
    result["guardrail_match"] = result["guardrail_match"].fillna(0).astype(int)
    return result


def _largest_component(assignments: pd.DataFrame) -> int:
    return int(assignments["predicted_group_id"].value_counts().max())


def run_pipeline(
    output_root: Path = PROJECT_ROOT,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Generate observations, select a threshold, evaluate, and create artifacts."""
    config = config or load_config()
    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    figures_dir = reports_dir / "figures"
    for path in (data_dir, reports_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)

    orders, truth = generate_orders(
        seed=config.seed,
        buildings=config.buildings,
        units_per_building=config.units_per_building,
        validation_share=config.validation_share,
        test_share=config.test_share,
    )
    validation = validate_inputs(orders, truth)
    signatures = build_account_signatures(orders)
    candidates = generate_candidate_pairs(signatures)
    pairs = build_pair_features(candidates, signatures, truth)

    validation_pairs = pairs[pairs["split"] == "validation"]
    selected_threshold, threshold_curve = select_threshold(
        validation_pairs,
        threshold_min=config.threshold_min,
        threshold_max=config.threshold_max,
        threshold_step=config.threshold_step,
    )
    test_pairs, pair_summary = evaluate_pair_models(
        pairs[pairs["split"] == "test"], selected_threshold
    )
    noise_recall = recall_by_noise(test_pairs)

    test_truth = truth[truth["split"] == "test"]
    test_accounts = sorted(test_truth["account_id"])
    baseline_assignments = component_assignments(
        test_accounts, test_pairs, prediction_column="baseline_match"
    )
    enhanced_assignments = component_assignments(
        test_accounts, test_pairs, prediction_column="enhanced_match"
    )
    guarded_assignments, test_guardrail_audit = guarded_component_assignments(
        test_accounts,
        test_pairs,
        prediction_column="enhanced_match",
        score_column="relation_score",
        max_component_size=config.max_component_size,
    )
    test_pairs = _mark_guarded_pairs(test_pairs, test_guardrail_audit)
    pair_summary["guarded_resolution"] = binary_metrics(
        test_pairs["is_related"], test_pairs["guardrail_match"]
    )
    cluster_scores = {
        "exact_key_baseline": cluster_ari(baseline_assignments, test_truth),
        "scored_resolution": cluster_ari(enhanced_assignments, test_truth),
        "guarded_resolution": cluster_ari(guarded_assignments, test_truth),
    }

    all_pairs = pairs.copy()
    all_pairs["enhanced_match"] = all_pairs["relation_score"].ge(selected_threshold).astype(int)
    all_accounts = sorted(truth["account_id"])
    unconstrained_all_assignments = component_assignments(
        all_accounts, all_pairs, prediction_column="enhanced_match"
    )
    all_assignments, component_guardrail = guarded_component_assignments(
        all_accounts,
        all_pairs,
        prediction_column="enhanced_match",
        score_column="relation_score",
        max_component_size=config.max_component_size,
    )
    all_pairs = _mark_guarded_pairs(all_pairs, component_guardrail)
    group_profiles = build_group_profiles(
        all_assignments, signatures, all_pairs, prediction_column="guardrail_match"
    )

    candidate_recall: dict[str, float] = {}
    for split in ("validation", "test"):
        expected = all_true_related_pairs(truth, split)
        observed = set(
            pairs.loc[pairs["split"] == split, ["account_a", "account_b"]].itertuples(
                index=False, name=None
            )
        )
        candidate_recall[split] = len(expected.intersection(observed)) / len(expected)

    fingerprint = _fingerprint(
        [
            orders,
            truth,
            signatures,
            pairs,
            threshold_curve,
            test_pairs,
            all_assignments,
            component_guardrail,
            group_profiles,
        ]
    )

    export_orders = orders.copy()
    export_orders["order_date"] = export_orders["order_date"].dt.date.astype(str)
    export_orders.to_csv(data_dir / "synthetic_orders.csv", index=False)
    truth.to_csv(data_dir / "synthetic_ground_truth.csv", index=False)
    signatures.to_csv(reports_dir / "account_signatures.csv", index=False)
    threshold_curve.to_csv(reports_dir / "threshold_curve.csv", index=False)
    test_pairs.to_csv(reports_dir / "test_pair_predictions.csv", index=False)
    component_guardrail.to_csv(reports_dir / "component_guardrail.csv", index=False)
    noise_recall.to_csv(reports_dir / "noise_recall.csv", index=False)
    all_assignments.to_csv(reports_dir / "predicted_group_assignments.csv", index=False)
    group_profiles.to_csv(reports_dir / "review_groups.csv", index=False)

    metrics: dict[str, Any] = {
        "data": asdict(validation),
        "candidate_recall": candidate_recall,
        "selected_threshold": selected_threshold,
        "pair_evaluation": pair_summary,
        "cluster_ari": cluster_scores,
        "component_guardrail": {
            "max_component_size": config.max_component_size,
            "test_deferred_edges": int(
                test_guardrail_audit["guardrail_decision"].eq("deferred_component_cap").sum()
            ),
            "all_deferred_edges": int(
                component_guardrail["guardrail_decision"].eq("deferred_component_cap").sum()
            ),
            "largest_test_component_before": _largest_component(enhanced_assignments),
            "largest_test_component_after": _largest_component(guarded_assignments),
            "largest_all_component_before": _largest_component(unconstrained_all_assignments),
            "largest_all_component_after": _largest_component(all_assignments),
            "deferred_edges_are_review_candidates": True,
        },
        "artifact_fingerprint": fingerprint,
        "evaluation_boundary": "Synthetic address observations and held-out buildings only",
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )

    plot_model_comparison(pair_summary, cluster_scores, figures_dir / "model_comparison.png")
    plot_threshold_curve(
        threshold_curve, selected_threshold, figures_dir / "threshold_selection.png"
    )
    plot_noise_recall(noise_recall, figures_dir / "noise_recall.png")
    plot_review_graph(test_pairs, enhanced_assignments, figures_dir / "review_graph.png")
    _write_summary(metrics, reports_dir / "run_summary.md")
    return metrics


def _write_summary(metrics: dict[str, Any], output_path: Path) -> None:
    baseline = metrics["pair_evaluation"]["exact_key_baseline"]
    enhanced = metrics["pair_evaluation"]["scored_resolution"]
    text = f"""# Reproduction summary

- Selected validation threshold: {metrics["selected_threshold"]:.2f}
- Exact-key baseline F1 on test buildings: {baseline["f1"]:.3f}
- Scored-resolution F1 on test buildings: {enhanced["f1"]:.3f}
- Guarded-resolution F1 on test buildings: {metrics["pair_evaluation"]["guarded_resolution"]["f1"]:.3f}
- Exact-key cluster ARI: {metrics["cluster_ari"]["exact_key_baseline"]:.3f}
- Scored-resolution cluster ARI: {metrics["cluster_ari"]["scored_resolution"]:.3f}
- Guarded-resolution cluster ARI: {metrics["cluster_ari"]["guarded_resolution"]:.3f}
- Deferred component-expansion edges: {metrics["component_guardrail"]["all_deferred_edges"]}
- Largest component before / after guardrail: {metrics["component_guardrail"]["largest_all_component_before"]} / {metrics["component_guardrail"]["largest_all_component_after"]}
- Artifact fingerprint: `{metrics["artifact_fingerprint"]}`

These metrics describe a fully synthetic entity-resolution benchmark. Predicted groups require review and do not prove family, fraud, or legal relationships.
"""
    output_path.write_text(text, encoding="utf-8")
