"""Candidate generation and shared-address pair scoring."""

from __future__ import annotations

from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

FEATURE_WEIGHTS = {
    "city_match": 0.10,
    "street_similarity": 0.30,
    "building_match": 0.20,
    "unit_match": 0.30,
    "postcode_match": 0.08,
    "family_token_match": 0.02,
}


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def generate_candidate_pairs(signatures: pd.DataFrame) -> pd.DataFrame:
    """Block on normalized city and building, then enumerate account pairs."""
    records: list[dict[str, str]] = []
    grouped = signatures.groupby(["city_norm", "building_norm"], dropna=False)
    for (city, building), group in grouped:
        accounts = sorted(group["account_id"].unique())
        for left, right in combinations(accounts, 2):
            records.append(
                {
                    "account_a": left,
                    "account_b": right,
                    "block_city": str(city),
                    "block_building": str(building),
                }
            )
    return pd.DataFrame.from_records(records)


def build_pair_features(
    candidates: pd.DataFrame,
    signatures: pd.DataFrame,
    truth: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate normalized pair features and attach labels only for evaluation."""
    left = signatures.add_suffix("_a").rename(columns={"account_id_a": "account_a"})
    right = signatures.add_suffix("_b").rename(columns={"account_id_b": "account_b"})
    frame = candidates.merge(left, on="account_a", how="left").merge(
        right, on="account_b", how="left"
    )
    frame["city_match"] = (frame["city_norm_a"] == frame["city_norm_b"]).astype(float)
    frame["street_similarity"] = [
        _similarity(left_value, right_value)
        for left_value, right_value in zip(
            frame["street_norm_a"], frame["street_norm_b"], strict=True
        )
    ]
    frame["building_match"] = (frame["building_norm_a"] == frame["building_norm_b"]).astype(float)
    frame["unit_match"] = (
        frame["unit_norm_a"].ne("")
        & frame["unit_norm_b"].ne("")
        & (frame["unit_norm_a"] == frame["unit_norm_b"])
    ).astype(float)
    frame["postcode_available"] = frame["postcode_norm_a"].ne("") & frame["postcode_norm_b"].ne("")
    frame["postcode_match"] = (
        frame["postcode_available"] & (frame["postcode_norm_a"] == frame["postcode_norm_b"])
    ).astype(float)
    frame["family_token_available"] = frame["family_token_norm_a"].ne("") & frame[
        "family_token_norm_b"
    ].ne("")
    frame["family_token_match"] = (
        frame["family_token_available"]
        & (frame["family_token_norm_a"] == frame["family_token_norm_b"])
    ).astype(float)

    numerator = pd.Series(0.0, index=frame.index)
    denominator = pd.Series(0.0, index=frame.index)
    for feature, weight in FEATURE_WEIGHTS.items():
        if feature == "postcode_match":
            available = frame["postcode_available"].astype(float)
        elif feature == "family_token_match":
            available = frame["family_token_available"].astype(float)
        else:
            available = 1.0
        numerator += frame[feature] * weight
        denominator += available * weight
    frame["relation_score"] = numerator / denominator
    frame["baseline_match"] = (
        frame["exact_address_key_a"].ne("")
        & (frame["exact_address_key_a"] == frame["exact_address_key_b"])
    ).astype(int)

    truth_left = truth.add_suffix("_a").rename(columns={"account_id_a": "account_a"})
    truth_right = truth.add_suffix("_b").rename(columns={"account_id_b": "account_b"})
    frame = frame.merge(truth_left, on="account_a", how="left").merge(
        truth_right, on="account_b", how="left"
    )
    if not (frame["split_a"] == frame["split_b"]).all():
        raise ValueError("Candidate pairs must not cross building-level splits")
    frame["split"] = frame["split_a"]
    frame["building_id"] = frame["building_id_a"]
    frame["is_related"] = (
        (frame["true_group_id_a"] == frame["true_group_id_b"])
        & (frame["account_a"] != frame["account_b"])
    ).astype(int)
    severity = {"clean": 0, "moderate": 1, "heavy": 2}
    reverse = {value: key for key, value in severity.items()}
    frame["pair_noise_level"] = [
        reverse[max(severity[left_value], severity[right_value])]
        for left_value, right_value in zip(
            frame["noise_level_a"], frame["noise_level_b"], strict=True
        )
    ]
    keep = [
        "account_a",
        "account_b",
        "split",
        "building_id",
        "pair_noise_level",
        *FEATURE_WEIGHTS,
        "postcode_available",
        "family_token_available",
        "relation_score",
        "baseline_match",
        "is_related",
    ]
    return frame[keep].sort_values(["account_a", "account_b"], ignore_index=True)


def all_true_related_pairs(truth: pd.DataFrame, split: str) -> set[tuple[str, str]]:
    """Enumerate every positive account pair in a split for blocking evaluation."""
    selected = truth[truth["split"] == split]
    pairs: set[tuple[str, str]] = set()
    for _, group in selected.groupby("true_group_id"):
        accounts = sorted(group["account_id"])
        pairs.update(combinations(accounts, 2))
    return pairs
