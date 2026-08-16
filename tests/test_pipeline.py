import json
from pathlib import Path

from customer_relation_detection.config import AnalysisConfig
from customer_relation_detection.pipeline import run_pipeline


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
        max_component_size=5,
    )


def test_pipeline_writes_required_artifacts(tmp_path: Path) -> None:
    metrics = run_pipeline(tmp_path, config=_small_config())
    required = [
        "data/synthetic_orders.csv",
        "data/synthetic_ground_truth.csv",
        "reports/metrics.json",
        "reports/test_pair_predictions.csv",
        "reports/component_guardrail.csv",
        "reports/review_groups.csv",
        "reports/figures/model_comparison.png",
        "reports/figures/threshold_selection.png",
        "reports/figures/noise_recall.png",
        "reports/figures/review_graph.png",
    ]
    assert all((tmp_path / path).exists() for path in required)
    assert metrics["candidate_recall"]["test"] == 1.0
    assert metrics["component_guardrail"]["largest_all_component_after"] <= 5


def test_core_artifacts_are_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_metrics = run_pipeline(first, config=_small_config())
    second_metrics = run_pipeline(second, config=_small_config())
    assert first_metrics["artifact_fingerprint"] == second_metrics["artifact_fingerprint"]
    assert json.loads((first / "reports/metrics.json").read_text()) == json.loads(
        (second / "reports/metrics.json").read_text()
    )
