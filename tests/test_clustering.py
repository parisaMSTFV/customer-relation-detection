import pandas as pd
import pytest

from customer_relation_detection.clustering import (
    build_group_profiles,
    cluster_ari,
    component_assignments,
    guarded_component_assignments,
)
from customer_relation_detection.evaluation import evaluate_pair_models, select_threshold
from customer_relation_detection.normalization import build_account_signatures
from customer_relation_detection.pairs import (
    attach_evaluation_labels,
    build_pair_features,
    generate_candidate_pairs,
)
from customer_relation_detection.synthetic import generate_orders


def test_connected_components_include_test_singletons() -> None:
    orders, truth = generate_orders(buildings=40)
    signatures = build_account_signatures(orders)
    features = build_pair_features(generate_candidate_pairs(signatures), signatures)
    pairs = attach_evaluation_labels(features, truth)
    threshold, _ = select_threshold(pairs[pairs["split"] == "validation"], 0.65, 0.95, 0.01)
    test_pairs, _ = evaluate_pair_models(pairs[pairs["split"] == "test"], threshold)
    test_truth = truth[truth["split"] == "test"]
    assignments = component_assignments(
        sorted(test_truth["account_id"]), test_pairs, "enhanced_match"
    )
    assert len(assignments) == len(test_truth)
    assert 0 <= cluster_ari(assignments, test_truth) <= 1


def test_component_guardrail_defers_transitive_expansion_deterministically() -> None:
    accounts = ["A", "B", "C", "D", "E", "F"]
    pairs = pd.DataFrame(
        {
            "account_a": ["A", "B", "C", "D", "E"],
            "account_b": ["B", "C", "D", "E", "F"],
            "relation_score": [0.99, 0.98, 0.97, 0.96, 0.95],
            "enhanced_match": [1, 1, 1, 1, 1],
        }
    )
    assignments, audit = guarded_component_assignments(
        accounts, pairs, "enhanced_match", "relation_score", max_component_size=3
    )
    assert assignments["predicted_group_id"].value_counts().max() == 3
    assert audit["guardrail_decision"].tolist() == [
        "accepted_merge",
        "accepted_merge",
        "deferred_component_cap",
        "accepted_merge",
        "accepted_merge",
    ]
    assert audit.loc[2, "proposed_component_size"] == 4


def test_component_builders_reject_self_pairs_and_invalid_scores() -> None:
    pairs = pd.DataFrame(
        {
            "account_a": ["A"],
            "account_b": ["A"],
            "relation_score": [1.2],
            "enhanced_match": [1],
        }
    )
    with pytest.raises(ValueError, match="scores"):
        guarded_component_assignments(["A"], pairs, "enhanced_match", "relation_score", 2)
    pairs["relation_score"] = 0.9
    with pytest.raises(ValueError, match="Self-pairs"):
        component_assignments(["A"], pairs, "enhanced_match")


def test_missing_family_tokens_do_not_create_a_household_signal() -> None:
    assignments = pd.DataFrame({"account_id": ["A", "B"], "predicted_group_id": ["P1", "P1"]})
    signatures = pd.DataFrame({"account_id": ["A", "B"], "family_token_norm": ["", ""]})
    pairs = pd.DataFrame(
        {
            "account_a": ["A"],
            "account_b": ["B"],
            "relation_score": [0.9],
            "guardrail_match": [1],
        }
    )
    profile = build_group_profiles(assignments, signatures, pairs, "guardrail_match").iloc[0]
    assert profile["dominant_family_token_share"] == 0.0
    assert profile["signal_type"] == "shared_residence_signal"
