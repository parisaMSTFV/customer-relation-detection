"""Bounded candidate generation, evidence scoring, and evaluation labels."""

from __future__ import annotations

import hashlib
from difflib import SequenceMatcher
from itertools import combinations

import pandas as pd

SCORING_WEIGHTS = {
    "street_similarity": 0.45,
    "unit_match": 0.35,
    "postcode_match": 0.15,
    "family_token_match": 0.05,
}
FEATURE_COLUMNS = (
    "city_match",
    "street_similarity",
    "building_match",
    "unit_match",
    "postcode_match",
    "family_token_match",
)
BLOCK_RULES = {
    "city_building": ("city_norm", "building_norm"),
    "postcode": ("postcode_norm",),
    "city_street": ("city_norm", "street_norm"),
}


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(SequenceMatcher(None, left, right).ratio())


def _block_hash(strategy: str, values: tuple[object, ...]) -> str:
    material = strategy + "|" + "|".join(str(value) for value in values)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def generate_candidate_pairs(
    signatures: pd.DataFrame,
    *,
    max_block_size: int = 250,
) -> pd.DataFrame:
    """Generate deduplicated pairs across several bounded blocking passes.

    Oversized blocks are deferred instead of causing quadratic expansion. A
    privacy-minimized block audit is attached to ``DataFrame.attrs['block_audit']``.
    """
    if max_block_size < 2:
        raise ValueError("max_block_size must be at least two")
    required = {"account_id", *{column for columns in BLOCK_RULES.values() for column in columns}}
    missing = required.difference(signatures.columns)
    if missing:
        raise ValueError(f"Signatures are missing required columns: {sorted(missing)}")
    if signatures["account_id"].duplicated().any():
        raise ValueError("Signatures must have one row per account")

    pair_sources: dict[tuple[str, str], set[str]] = {}
    audit_rows: list[dict[str, object]] = []
    for strategy, columns in BLOCK_RULES.items():
        eligible = signatures[list(columns)].ne("").all(axis=1)
        for key, group in signatures[eligible].groupby(list(columns), dropna=False, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            accounts = sorted(group["account_id"].astype(str).unique())
            if len(accounts) < 2:
                continue
            oversized = len(accounts) > max_block_size
            generated_pairs = 0 if oversized else len(accounts) * (len(accounts) - 1) // 2
            audit_rows.append(
                {
                    "blocking_strategy": strategy,
                    "block_key_hash": _block_hash(strategy, values),
                    "block_size": len(accounts),
                    "generated_pairs": generated_pairs,
                    "block_decision": "deferred_oversized" if oversized else "accepted",
                }
            )
            if oversized:
                continue
            for account_a, account_b in combinations(accounts, 2):
                pair_sources.setdefault((account_a, account_b), set()).add(strategy)

    records = [
        {
            "account_a": account_a,
            "account_b": account_b,
            "blocking_strategies": "+".join(sorted(strategies)),
            "blocking_passes": len(strategies),
        }
        for (account_a, account_b), strategies in sorted(pair_sources.items())
    ]
    candidates = pd.DataFrame.from_records(
        records,
        columns=["account_a", "account_b", "blocking_strategies", "blocking_passes"],
    )
    candidates.attrs["block_audit"] = pd.DataFrame.from_records(
        audit_rows,
        columns=[
            "blocking_strategy",
            "block_key_hash",
            "block_size",
            "generated_pairs",
            "block_decision",
        ],
    )
    return candidates


def build_pair_features(candidates: pd.DataFrame, signatures: pd.DataFrame) -> pd.DataFrame:
    """Calculate evidence and scores without loading or requiring ground truth."""
    required = {"account_a", "account_b", "blocking_strategies", "blocking_passes"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Candidates are missing required columns: {sorted(missing)}")
    left = signatures.add_suffix("_a").rename(columns={"account_id_a": "account_a"})
    right = signatures.add_suffix("_b").rename(columns={"account_id_b": "account_b"})
    frame = candidates.merge(left, on="account_a", how="left", validate="many_to_one").merge(
        right, on="account_b", how="left", validate="many_to_one"
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
    for feature, weight in SCORING_WEIGHTS.items():
        if feature == "postcode_match":
            available: float | pd.Series = frame["postcode_available"].astype(float)
        elif feature == "family_token_match":
            available = frame["family_token_available"].astype(float)
        else:
            available = 1.0
        numerator += frame[feature] * weight
        denominator += available * weight
    frame["relation_score"] = (numerator / denominator).fillna(0.0)
    frame["baseline_match"] = (
        frame["exact_address_key_a"].ne("")
        & (frame["exact_address_key_a"] == frame["exact_address_key_b"])
    ).astype(int)
    keep = [
        "account_a",
        "account_b",
        "blocking_strategies",
        "blocking_passes",
        *FEATURE_COLUMNS,
        "postcode_available",
        "family_token_available",
        "relation_score",
        "baseline_match",
    ]
    return frame[keep].sort_values(["account_a", "account_b"], ignore_index=True)


def attach_evaluation_labels(pair_features: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Attach synthetic labels in a separate, evaluation-only operation."""
    truth_left = truth.add_suffix("_a").rename(columns={"account_id_a": "account_a"})
    truth_right = truth.add_suffix("_b").rename(columns={"account_id_b": "account_b"})
    frame = pair_features.merge(
        truth_left, on="account_a", how="left", validate="many_to_one"
    ).merge(truth_right, on="account_b", how="left", validate="many_to_one")
    if frame[["split_a", "split_b"]].isna().any().any():
        raise ValueError("Every candidate account must exist in ground truth")
    frame = frame[frame["split_a"] == frame["split_b"]].copy()
    frame["split"] = frame["split_a"]
    frame["building_id"] = frame["building_id_a"]
    frame["is_related"] = (
        (frame["true_group_id_a"] == frame["true_group_id_b"])
        & (frame["account_a"] != frame["account_b"])
    ).astype(int)
    severity = {"clean": 0, "moderate": 1, "heavy": 2}
    reverse = {value: key for key, value in severity.items()}
    try:
        frame["pair_noise_level"] = [
            reverse[max(severity[left_value], severity[right_value])]
            for left_value, right_value in zip(
                frame["noise_level_a"], frame["noise_level_b"], strict=True
            )
        ]
    except KeyError as error:
        raise ValueError(f"Unsupported noise level: {error.args[0]}") from error
    keep = [
        "account_a",
        "account_b",
        "split",
        "building_id",
        "pair_noise_level",
        "blocking_strategies",
        "blocking_passes",
        *FEATURE_COLUMNS,
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
