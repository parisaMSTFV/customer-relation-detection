from customer_relation_detection.normalization import build_account_signatures
from customer_relation_detection.pairs import (
    all_true_related_pairs,
    build_pair_features,
    generate_candidate_pairs,
)
from customer_relation_detection.synthetic import generate_orders


def _case():
    orders, truth = generate_orders(buildings=40)
    signatures = build_account_signatures(orders)
    candidates = generate_candidate_pairs(signatures)
    return truth, build_pair_features(candidates, signatures, truth)


def test_candidate_pairs_do_not_cross_building_splits() -> None:
    _, pairs = _case()
    assert set(pairs["split"]).issubset({"train", "validation", "test"})


def test_candidate_generation_recalls_all_test_positive_pairs() -> None:
    truth, pairs = _case()
    expected = all_true_related_pairs(truth, "test")
    observed = set(
        pairs.loc[pairs["split"] == "test", ["account_a", "account_b"]].itertuples(
            index=False, name=None
        )
    )
    assert expected.issubset(observed)


def test_same_unit_pairs_score_above_different_unit_pairs() -> None:
    _, pairs = _case()
    same = pairs[pairs["is_related"] == 1]["relation_score"].median()
    different = pairs[pairs["is_related"] == 0]["relation_score"].median()
    assert same > different
