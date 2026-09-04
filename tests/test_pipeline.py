import json
from dataclasses import replace
from pathlib import Path

from customer_relation_detection.config import AnalysisConfig
from customer_relation_detection.pipeline import (
    evaluate_policy_stability,
    run_analysis,
    run_pipeline,
)
from customer_relation_detection.synthetic import generate_orders


def _small_config() -> AnalysisConfig:
    return AnalysisConfig(
        seed=42,
        buildings=30,
        units_per_building=4,
        validation_share=0.2,
        test_share=0.2,
        threshold_min=0.65,
        threshold_max=0.95,
        threshold_step=0.01,
        max_component_size=8,
        component_size_candidates=(4, 6, 8),
        stability_seeds=(),
    )


def test_pipeline_writes_required_artifacts(tmp_path: Path) -> None:
    metrics = run_pipeline(tmp_path, config=_small_config())
    required = [
        "data/synthetic_orders.csv",
        "data/synthetic_ground_truth.csv",
        "reports/metrics.json",
        "reports/candidate_block_audit.csv",
        "reports/component_policy_curve.csv",
        "reports/policy_stability.csv",
        "reports/subgroup_errors.csv",
        "reports/test_pair_predictions.csv",
        "reports/component_guardrail.csv",
        "reports/review_groups.csv",
        "reports/figures/model_comparison.png",
        "reports/figures/threshold_selection.png",
        "reports/figures/noise_recall.png",
        "reports/figures/review_graph.png",
    ]
    assert all((tmp_path / path).exists() for path in required)
    assert metrics["blocking"]["candidate_recall"]["test"] == 1.0
    assert metrics["component_guardrail"]["largest_all_component_after"] <= 8
    assert metrics["policy"]["selection_split"] == "validation"


def test_core_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_metrics = run_pipeline(first, config=_small_config())
    second_metrics = run_pipeline(second, config=_small_config())
    assert first_metrics["artifact_fingerprint"] == second_metrics["artifact_fingerprint"]
    assert json.loads((first / "reports/metrics.json").read_text()) == json.loads(
        (second / "reports/metrics.json").read_text()
    )


def test_operational_analysis_exports_only_pseudonymized_review_data(tmp_path: Path) -> None:
    orders, _ = generate_orders(buildings=30)
    raw_account = str(orders.loc[0, "account_id"])
    raw_city = str(orders.loc[0, "city"])
    metrics = run_analysis(
        orders,
        tmp_path,
        identifier_salt="test-only-secret-salt",
        config=_small_config(),
    )
    exported = "\n".join(
        path.read_text(encoding="utf-8") for path in (tmp_path / "reports").glob("*")
    )
    assert raw_account not in exported
    assert raw_city not in exported
    assert metrics["privacy"]["raw_addresses_exported"] is False


def test_frozen_policy_is_measured_across_configured_seeds() -> None:
    config = replace(_small_config(), stability_seeds=(7,), stability_buildings=30)
    stability = evaluate_policy_stability(config)
    assert stability["seed"].tolist() == [7]
    assert stability["test_candidate_recall"].eq(1.0).all()
    assert stability["guarded_component_f1"].between(0, 1).all()
