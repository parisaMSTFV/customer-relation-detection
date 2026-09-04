import pandas as pd

from customer_relation_detection.normalization import build_account_signatures
from customer_relation_detection.pairs import (
    SCORING_WEIGHTS,
    all_true_related_pairs,
    attach_evaluation_labels,
    build_pair_features,
    generate_candidate_pairs,
)
from customer_relation_detection.synthetic import generate_orders


def _case():
    orders, truth = generate_orders(buildings=40)
    signatures = build_account_signatures(orders)
    candidates = generate_candidate_pairs(signatures)
    features = build_pair_features(candidates, signatures)
    return truth, attach_evaluation_labels(features, truth)


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


def test_block_keys_are_not_score_weights() -> None:
    assert "city_match" not in SCORING_WEIGHTS
    assert "building_match" not in SCORING_WEIGHTS


def test_multi_pass_blocking_recovers_a_building_typo() -> None:
    signatures = pd.DataFrame(
        {
            "account_id": ["A", "B"],
            "city_norm": ["تهران", "تهران"],
            "street_norm": ["ولیعصر", "ولیعصر"],
            "building_norm": ["12", "13"],
            "postcode_norm": ["", ""],
        }
    )
    candidates = generate_candidate_pairs(signatures)
    assert len(candidates) == 1
    assert candidates.loc[0, "blocking_strategies"] == "city_street"


def test_oversized_blocks_are_deferred_without_raw_keys() -> None:
    signatures = pd.DataFrame(
        {
            "account_id": ["A", "B", "C"],
            "city_norm": ["c1", "c2", "c3"],
            "street_norm": ["s1", "s2", "s3"],
            "building_norm": ["1", "2", "3"],
            "postcode_norm": ["shared", "shared", "shared"],
        }
    )
    candidates = generate_candidate_pairs(signatures, max_block_size=2)
    assert candidates.empty
    audit = candidates.attrs["block_audit"]
    assert audit.loc[0, "block_decision"] == "deferred_oversized"
    assert "shared" not in audit.to_csv(index=False)
