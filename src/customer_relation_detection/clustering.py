"""Convert predicted relation pairs into reviewable account groups."""

from __future__ import annotations

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def component_assignments(
    accounts: list[str],
    pairs: pd.DataFrame,
    prediction_column: str,
) -> pd.DataFrame:
    """Create stable connected-component labels including singleton accounts."""
    graph = nx.Graph()
    graph.add_nodes_from(sorted(accounts))
    predicted = pairs[pairs[prediction_column] == 1]
    graph.add_edges_from(predicted[["account_a", "account_b"]].itertuples(index=False, name=None))
    components = sorted(
        (sorted(component) for component in nx.connected_components(graph)), key=lambda x: x[0]
    )
    records: list[dict[str, str]] = []
    for position, component in enumerate(components, start=1):
        group_id = f"PRED-{position:05d}"
        records.extend(
            {"account_id": account, "predicted_group_id": group_id} for account in component
        )
    return pd.DataFrame.from_records(records).sort_values("account_id", ignore_index=True)


def cluster_ari(assignments: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Evaluate predicted connected components against true account groups."""
    merged = truth[["account_id", "true_group_id"]].merge(assignments, on="account_id")
    return float(adjusted_rand_score(merged["true_group_id"], merged["predicted_group_id"]))


def build_group_profiles(
    assignments: pd.DataFrame,
    signatures: pd.DataFrame,
    predicted_pairs: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize evidence without claiming legal, familial, or fraud relationships."""
    member_data = assignments.merge(
        signatures[["account_id", "family_token_norm"]], on="account_id", how="left"
    )
    rows: list[dict[str, object]] = []
    positive_pairs = predicted_pairs[predicted_pairs["enhanced_match"] == 1]
    for group_id, group in member_data.groupby("predicted_group_id"):
        members = sorted(group["account_id"])
        scores = positive_pairs[
            positive_pairs["account_a"].isin(members) & positive_pairs["account_b"].isin(members)
        ]["relation_score"]
        family_share = float(group["family_token_norm"].value_counts(normalize=True).max())
        if len(members) == 1:
            signal = "singleton"
        elif len(members) <= 5 and family_share >= 0.60:
            signal = "shared_household_signal"
        elif len(members) <= 5:
            signal = "shared_residence_signal"
        else:
            signal = "shared_location_review"
        rows.append(
            {
                "predicted_group_id": group_id,
                "members": len(members),
                "signal_type": signal,
                "dominant_family_token_share": family_share,
                "minimum_link_score": float(scores.min()) if not scores.empty else 0.0,
                "mean_link_score": float(scores.mean()) if not scores.empty else 0.0,
                "review_required": True,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("predicted_group_id", ignore_index=True)
