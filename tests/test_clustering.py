from customer_relation_detection.clustering import cluster_ari, component_assignments
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
