import pandas as pd

from customer_relation_detection.clustering import (
    cluster_ari,
    component_assignments,
    guarded_component_assignments,
)
from customer_relation_detection.evaluation import evaluate_pair_models, select_threshold
from customer_relation_detection.normalization import build_account_signatures
from customer_relation_detection.pairs import build_pair_features, generate_candidate_pairs
from customer_relation_detection.synthetic import generate_orders


def test_connected_components_include_test_singletons() -> None:
    orders, truth = generate_orders(buildings=40)
    signatures = build_account_signatures(orders)
    pairs = build_pair_features(generate_candidate_pairs(signatures), signatures, truth)
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
