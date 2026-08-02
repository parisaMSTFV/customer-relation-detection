from customer_relation_detection.evaluation import (
    evaluate_pair_models,
    recall_by_noise,
    select_threshold,
)
from customer_relation_detection.normalization import build_account_signatures
from customer_relation_detection.pairs import build_pair_features, generate_candidate_pairs
from customer_relation_detection.synthetic import generate_orders


def _pairs():
    orders, truth = generate_orders(buildings=60)
    signatures = build_account_signatures(orders)
    return build_pair_features(generate_candidate_pairs(signatures), signatures, truth)


def test_threshold_is_selected_on_validation_and_applied_to_test() -> None:
    pairs = _pairs()
    threshold, curve = select_threshold(pairs[pairs["split"] == "validation"], 0.65, 0.95, 0.01)
    evaluated, summary = evaluate_pair_models(pairs[pairs["split"] == "test"], threshold)
    assert 0.65 <= threshold <= 0.95
    assert len(curve) == 31
    assert set(evaluated["enhanced_match"].unique()).issubset({0, 1})
    assert summary["scored_resolution"]["f1"] >= summary["exact_key_baseline"]["f1"]


def test_noise_breakdown_contains_positive_pair_counts() -> None:
    pairs = _pairs()
    threshold, _ = select_threshold(pairs[pairs["split"] == "validation"], 0.65, 0.95, 0.01)
    evaluated, _ = evaluate_pair_models(pairs[pairs["split"] == "test"], threshold)
    breakdown = recall_by_noise(evaluated)
    assert breakdown["positive_pairs"].gt(0).all()
    assert breakdown["enhanced_recall"].between(0, 1).all()
