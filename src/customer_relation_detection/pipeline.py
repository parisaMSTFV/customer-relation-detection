"""Benchmark and privacy-minimized operational relation-resolution pipelines."""

from __future__ import annotations

import hashlib
import hmac
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
from customer_relation_detection.config import AnalysisConfig, load_config
from customer_relation_detection.evaluation import (
    bcubed_metrics,
    binary_metrics,
    component_pair_metrics,
    error_metrics_by_noise,
    evaluate_pair_models,
    recall_by_noise,
    select_component_policy,
)
from customer_relation_detection.normalization import build_account_signatures
from customer_relation_detection.pairs import (
    all_true_related_pairs,
    attach_evaluation_labels,
    build_pair_features,
    generate_candidate_pairs,
)
from customer_relation_detection.schema import validate_inputs, validate_observations
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
    if assignments.empty:
        return 0
    return int(assignments["predicted_group_id"].value_counts().max())


def _prepare_observations(
    orders: pd.DataFrame,
    config: AnalysisConfig,
    *,
    snapshot_date: object | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signatures = build_account_signatures(
        orders,
        snapshot_date=snapshot_date,
        history_days=config.address_history_days,
    )
    candidates = generate_candidate_pairs(signatures, max_block_size=config.max_block_size)
    block_audit = candidates.attrs["block_audit"]
    pair_features = build_pair_features(candidates, signatures)
    return signatures, candidates, pair_features, block_audit


def _policy_plot_curve(policy_curve: pd.DataFrame) -> pd.DataFrame:
    display = (
        policy_curve.sort_values(
            ["threshold", "weighted_error_cost", "component_f1"],
            ascending=[True, True, False],
            kind="mergesort",
        )
        .groupby("threshold", as_index=False)
        .first()
    )
    return display.rename(
        columns={
            "component_precision": "precision",
            "component_recall": "recall",
            "component_f1": "f1",
        }
    )


def evaluate_policy_stability(config: AnalysisConfig) -> pd.DataFrame:
    """Evaluate the frozen policy across deterministic generator seeds."""
    columns = [
        "seed",
        "accounts",
        "candidate_pairs",
        "test_candidate_recall",
        "scored_pair_precision",
        "scored_pair_recall",
        "scored_pair_f1",
        "guarded_pair_precision",
        "guarded_pair_recall",
        "guarded_pair_f1",
        "scored_component_precision",
        "scored_component_recall",
        "scored_component_f1",
        "guarded_component_precision",
        "guarded_component_recall",
        "guarded_component_f1",
        "guarded_bcubed_f1",
        "deferred_edges",
    ]
    rows: list[dict[str, float | int]] = []
    for seed in config.stability_seeds:
        orders, truth = generate_orders(
            seed=seed,
            buildings=config.stability_buildings,
            units_per_building=config.units_per_building,
            validation_share=config.validation_share,
            test_share=config.test_share,
        )
        validate_inputs(orders, truth)
        signatures, candidates, pair_features, _ = _prepare_observations(orders, config)
        pairs = attach_evaluation_labels(pair_features, truth)
        test_truth = truth[truth["split"] == "test"]
        test_pairs, pair_summary = evaluate_pair_models(
            pairs[pairs["split"] == "test"], config.policy_threshold
        )
        accounts = sorted(test_truth["account_id"].astype(str))
        scored_assignments = component_assignments(accounts, test_pairs, "enhanced_match")
        guarded_assignments, guardrail = guarded_component_assignments(
            accounts,
            test_pairs,
            prediction_column="enhanced_match",
            score_column="relation_score",
            max_component_size=config.policy_max_component_size,
        )
        test_pairs = _mark_guarded_pairs(test_pairs, guardrail)
        guarded_pair = binary_metrics(test_pairs["is_related"], test_pairs["guardrail_match"])
        scored_component = component_pair_metrics(scored_assignments, test_truth)
        guarded_component = component_pair_metrics(guarded_assignments, test_truth)
        expected = all_true_related_pairs(truth, "test")
        observed = set(test_pairs[["account_a", "account_b"]].itertuples(index=False, name=None))
        scored_pair = pair_summary["scored_resolution"]
        rows.append(
            {
                "seed": seed,
                "accounts": len(truth),
                "candidate_pairs": len(candidates),
                "test_candidate_recall": (
                    len(expected.intersection(observed)) / len(expected) if expected else 1.0
                ),
                "scored_pair_precision": float(scored_pair["precision"]),
                "scored_pair_recall": float(scored_pair["recall"]),
                "scored_pair_f1": float(scored_pair["f1"]),
                "guarded_pair_precision": float(guarded_pair["precision"]),
                "guarded_pair_recall": float(guarded_pair["recall"]),
                "guarded_pair_f1": float(guarded_pair["f1"]),
                "scored_component_precision": float(scored_component["precision"]),
                "scored_component_recall": float(scored_component["recall"]),
                "scored_component_f1": float(scored_component["f1"]),
                "guarded_component_precision": float(guarded_component["precision"]),
                "guarded_component_recall": float(guarded_component["recall"]),
                "guarded_component_f1": float(guarded_component["f1"]),
                "guarded_bcubed_f1": bcubed_metrics(guarded_assignments, test_truth)["f1"],
                "deferred_edges": int(
                    guardrail["guardrail_decision"].eq("deferred_component_cap").sum()
                ),
            }
        )
    return pd.DataFrame.from_records(rows, columns=columns)


def _stability_summary(stability: pd.DataFrame) -> dict[str, Any]:
    if stability.empty:
        return {"seeds": 0}
    summary: dict[str, Any] = {
        "seeds": len(stability),
        "seed_values": stability["seed"].astype(int).tolist(),
        "buildings_per_seed": int(stability.attrs["buildings_per_seed"]),
    }
    for metric in (
        "guarded_pair_precision",
        "guarded_pair_recall",
        "guarded_pair_f1",
        "guarded_component_precision",
        "guarded_component_recall",
        "guarded_component_f1",
        "guarded_bcubed_f1",
    ):
        summary[metric] = {
            "mean": float(stability[metric].mean()),
            "std": float(stability[metric].std(ddof=0)),
            "min": float(stability[metric].min()),
            "max": float(stability[metric].max()),
        }
    return summary


def run_pipeline(
    output_root: Path | None = None,
    config: AnalysisConfig | None = None,
) -> dict[str, Any]:
    """Generate the synthetic benchmark, select a graph policy, and create artifacts."""
    from customer_relation_detection.reporting import (
        plot_model_comparison,
        plot_noise_recall,
        plot_review_graph,
        plot_threshold_curve,
    )

    output_root = Path.cwd() if output_root is None else Path(output_root)
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
    signatures, candidates, pair_features, block_audit = _prepare_observations(orders, config)
    pairs = attach_evaluation_labels(pair_features, truth)

    validation_truth = truth[truth["split"] == "validation"]
    validation_pairs = pairs[pairs["split"] == "validation"]
    selected_threshold, selected_component_size, policy_curve = select_component_policy(
        validation_pairs,
        validation_truth,
        threshold_min=config.threshold_min,
        threshold_max=config.threshold_max,
        threshold_step=config.threshold_step,
        component_sizes=config.component_size_candidates,
        precision_floor=config.precision_floor,
        false_merge_cost=config.false_merge_cost,
        false_split_cost=config.false_split_cost,
    )

    test_pairs, pair_summary = evaluate_pair_models(
        pairs[pairs["split"] == "test"], selected_threshold
    )
    test_truth = truth[truth["split"] == "test"]
    test_accounts = sorted(test_truth["account_id"].astype(str))
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
        max_component_size=selected_component_size,
    )
    test_pairs = _mark_guarded_pairs(test_pairs, test_guardrail_audit)
    pair_summary["guarded_resolution"] = binary_metrics(
        test_pairs["is_related"], test_pairs["guardrail_match"]
    )
    noise_recall = recall_by_noise(test_pairs)
    subgroup_errors = error_metrics_by_noise(test_pairs)
    assignments_by_model = {
        "exact_key_baseline": baseline_assignments,
        "scored_resolution": enhanced_assignments,
        "guarded_resolution": guarded_assignments,
    }
    cluster_scores = {
        model: cluster_ari(assignments, test_truth)
        for model, assignments in assignments_by_model.items()
    }
    component_scores = {
        model: component_pair_metrics(assignments, test_truth)
        for model, assignments in assignments_by_model.items()
    }
    bcubed_scores = {
        model: bcubed_metrics(assignments, test_truth)
        for model, assignments in assignments_by_model.items()
    }

    all_pairs = pairs.copy()
    all_pairs["enhanced_match"] = all_pairs["relation_score"].ge(selected_threshold).astype(int)
    all_accounts = sorted(truth["account_id"].astype(str))
    unconstrained_all_assignments = component_assignments(
        all_accounts, all_pairs, prediction_column="enhanced_match"
    )
    all_assignments, component_guardrail = guarded_component_assignments(
        all_accounts,
        all_pairs,
        prediction_column="enhanced_match",
        score_column="relation_score",
        max_component_size=selected_component_size,
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
        candidate_recall[split] = (
            len(expected.intersection(observed)) / len(expected) if expected else 1.0
        )
    policy_stability = evaluate_policy_stability(config)
    policy_stability.attrs["buildings_per_seed"] = config.stability_buildings

    fingerprint = _fingerprint(
        [
            orders,
            truth,
            signatures,
            candidates,
            pairs,
            block_audit,
            policy_curve,
            test_pairs,
            all_assignments,
            component_guardrail,
            group_profiles,
            policy_stability,
        ]
    )

    export_orders = orders.copy()
    export_orders["order_date"] = export_orders["order_date"].dt.date.astype(str)
    export_orders.to_csv(data_dir / "synthetic_orders.csv", index=False)
    truth.to_csv(data_dir / "synthetic_ground_truth.csv", index=False)
    signatures.to_csv(reports_dir / "account_signatures.csv", index=False)
    policy_curve.to_csv(reports_dir / "component_policy_curve.csv", index=False)
    _policy_plot_curve(policy_curve).to_csv(reports_dir / "threshold_curve.csv", index=False)
    block_audit.to_csv(reports_dir / "candidate_block_audit.csv", index=False)
    test_pairs.to_csv(reports_dir / "test_pair_predictions.csv", index=False)
    component_guardrail.to_csv(reports_dir / "component_guardrail.csv", index=False)
    noise_recall.to_csv(reports_dir / "noise_recall.csv", index=False)
    subgroup_errors.to_csv(reports_dir / "subgroup_errors.csv", index=False)
    policy_stability.to_csv(reports_dir / "policy_stability.csv", index=False)
    all_assignments.to_csv(reports_dir / "predicted_group_assignments.csv", index=False)
    group_profiles.to_csv(reports_dir / "review_groups.csv", index=False)

    selected_policy = policy_curve[policy_curve["selected"]].iloc[0]
    metrics: dict[str, Any] = {
        "data": asdict(validation),
        "blocking": {
            "strategies": ["city_building", "postcode", "city_street"],
            "candidate_pairs": len(candidates),
            "max_block_size": config.max_block_size,
            "deferred_oversized_blocks": int(
                block_audit["block_decision"].eq("deferred_oversized").sum()
            ),
            "candidate_recall": candidate_recall,
        },
        "policy": {
            "version": config.policy_version,
            "selected_threshold": selected_threshold,
            "selected_max_component_size": selected_component_size,
            "selection_split": "validation",
            "precision_floor": config.precision_floor,
            "false_merge_cost": config.false_merge_cost,
            "false_split_cost": config.false_split_cost,
            "validation_weighted_error_cost": float(selected_policy["weighted_error_cost"]),
            "validation_component_precision": float(selected_policy["component_precision"]),
            "validation_component_recall": float(selected_policy["component_recall"]),
        },
        "selected_threshold": selected_threshold,
        "pair_evaluation": pair_summary,
        "cluster_ari": cluster_scores,
        "component_pair_evaluation": component_scores,
        "bcubed_evaluation": bcubed_scores,
        "policy_stability": _stability_summary(policy_stability),
        "component_guardrail": {
            "max_component_size": selected_component_size,
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
        _policy_plot_curve(policy_curve),
        selected_threshold,
        figures_dir / "threshold_selection.png",
    )
    plot_noise_recall(noise_recall, figures_dir / "noise_recall.png")
    plot_review_graph(
        test_pairs,
        guarded_assignments,
        figures_dir / "review_graph.png",
        prediction_column="guardrail_match",
    )
    _write_summary(metrics, reports_dir / "run_summary.md")
    return metrics


def _account_reference(account_id: object, identifier_salt: str) -> str:
    digest = hmac.new(
        identifier_salt.encode("utf-8"), str(account_id).encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"REF-{digest[:20]}"


def run_analysis(
    orders: pd.DataFrame,
    output_root: Path,
    *,
    identifier_salt: str,
    config: AnalysisConfig | None = None,
    snapshot_date: object | None = None,
) -> dict[str, Any]:
    """Score unlabeled observations and export only pseudonymized review artifacts."""
    if len(identifier_salt) < 16:
        raise ValueError("identifier_salt must contain at least 16 characters")
    config = config or load_config()
    observation_summary = validate_observations(orders)
    signatures, candidates, pair_features, block_audit = _prepare_observations(
        orders, config, snapshot_date=snapshot_date
    )
    scored = pair_features.copy()
    scored["enhanced_match"] = scored["relation_score"].ge(config.policy_threshold).astype(int)
    accounts = sorted(signatures["account_id"].astype(str))
    assignments, guardrail = guarded_component_assignments(
        accounts,
        scored,
        prediction_column="enhanced_match",
        score_column="relation_score",
        max_component_size=config.policy_max_component_size,
    )
    scored = _mark_guarded_pairs(scored, guardrail)
    profiles = build_group_profiles(
        assignments, signatures, scored, prediction_column="guardrail_match"
    )

    reference_map = {
        account_id: _account_reference(account_id, identifier_salt) for account_id in accounts
    }
    safe_assignments = assignments.rename(columns={"account_id": "account_ref"}).copy()
    safe_assignments["account_ref"] = safe_assignments["account_ref"].map(reference_map)
    safe_guardrail = guardrail.copy()
    safe_guardrail["account_a"] = safe_guardrail["account_a"].map(reference_map)
    safe_guardrail["account_b"] = safe_guardrail["account_b"].map(reference_map)
    safe_guardrail = safe_guardrail.rename(
        columns={"account_a": "account_ref_a", "account_b": "account_ref_b"}
    )

    reports_dir = Path(output_root) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    safe_assignments.to_csv(reports_dir / "predicted_group_assignments.csv", index=False)
    safe_guardrail.to_csv(reports_dir / "component_guardrail.csv", index=False)
    profiles.to_csv(reports_dir / "review_groups.csv", index=False)
    block_audit.to_csv(reports_dir / "candidate_block_audit.csv", index=False)
    metadata: dict[str, Any] = {
        "observations": asdict(observation_summary),
        "policy": {
            "version": config.policy_version,
            "threshold": config.policy_threshold,
            "max_component_size": config.policy_max_component_size,
        },
        "blocking": {
            "candidate_pairs": len(candidates),
            "max_block_size": config.max_block_size,
            "deferred_oversized_blocks": int(
                block_audit["block_decision"].eq("deferred_oversized").sum()
            ),
        },
        "review": {
            "predicted_groups": len(profiles),
            "deferred_component_edges": int(
                guardrail["guardrail_decision"].eq("deferred_component_cap").sum()
            ),
        },
        "privacy": {
            "raw_addresses_exported": False,
            "raw_account_ids_exported": False,
            "identifier_method": "HMAC-SHA256 truncated to 20 hex characters",
            "purpose": "human review of shared-address signals",
            "retention": "Delete inputs and review artifacts according to the approved case policy.",
            "access": "Restrict artifacts to authorized reviewers; do not use for adverse decisions.",
        },
        "artifact_fingerprint": _fingerprint(
            [safe_assignments, safe_guardrail, profiles, block_audit]
        ),
    }
    (reports_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metadata


def _write_summary(metrics: dict[str, Any], output_path: Path) -> None:
    baseline = metrics["pair_evaluation"]["exact_key_baseline"]
    enhanced = metrics["pair_evaluation"]["scored_resolution"]
    text = f"""# Reproduction summary

- Selected validation threshold: {metrics["selected_threshold"]:.2f}
- Selected component-size cap: {metrics["policy"]["selected_max_component_size"]}
- Exact-key baseline F1 on test buildings: {baseline["f1"]:.3f}
- Scored-resolution F1 on test buildings: {enhanced["f1"]:.3f}
- Guarded-resolution F1 on test buildings: {metrics["pair_evaluation"]["guarded_resolution"]["f1"]:.3f}
- Exact-key cluster ARI: {metrics["cluster_ari"]["exact_key_baseline"]:.3f}
- Scored-resolution cluster ARI: {metrics["cluster_ari"]["scored_resolution"]:.3f}
- Guarded-resolution cluster ARI: {metrics["cluster_ari"]["guarded_resolution"]:.3f}
- Guarded B-cubed F1: {metrics["bcubed_evaluation"]["guarded_resolution"]["f1"]:.3f}
- Deferred component-expansion edges: {metrics["component_guardrail"]["all_deferred_edges"]}
- Largest component before / after guardrail: {metrics["component_guardrail"]["largest_all_component_before"]} / {metrics["component_guardrail"]["largest_all_component_after"]}
- Artifact fingerprint: `{metrics["artifact_fingerprint"]}`

These metrics describe a fully synthetic entity-resolution benchmark. Predicted groups require review and do not prove family, fraud, or legal relationships.
"""
    output_path.write_text(text, encoding="utf-8")
