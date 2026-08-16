"""Convert predicted relation pairs into reviewable account groups."""

from __future__ import annotations

import networkx as nx
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def guarded_component_assignments(
    accounts: list[str],
    pairs: pd.DataFrame,
    prediction_column: str,
    score_column: str,
    max_component_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build components while deferring edges that would exceed a size cap.

    Candidate links are considered from strongest to weakest with stable account-ID
    tie-breaking. A deferred link remains in the audit output; it is not silently
    converted into evidence that the accounts are unrelated.
    """
    if max_component_size < 2:
        raise ValueError("max_component_size must be at least two")
    account_ids = sorted(set(accounts))
    required = {"account_a", "account_b", prediction_column, score_column}
    missing = required.difference(pairs.columns)
    if missing:
        raise ValueError(f"Pairs are missing required columns: {sorted(missing)}")

    parent = {account: account for account in account_ids}
    sizes = {account: 1 for account in account_ids}

    def find(account: str) -> str:
        root = account
        while parent[root] != root:
            root = parent[root]
        while parent[account] != account:
            next_account = parent[account]
            parent[account] = root
            account = next_account
        return root

    active = pairs[pairs[prediction_column] == 1].copy()
    active = active.sort_values(
        [score_column, "account_a", "account_b"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    decisions: list[dict[str, object]] = []
    for row in active.itertuples(index=False):
        account_a = str(row.account_a)
        account_b = str(row.account_b)
        if account_a not in parent or account_b not in parent:
            raise ValueError("Predicted pairs contain accounts outside the supplied account list")
        root_a = find(account_a)
        root_b = find(account_b)
        size_a = sizes[root_a]
        size_b = sizes[root_b]
        if root_a == root_b:
            proposed_size = size_a
            decision = "accepted_redundant"
        else:
            proposed_size = size_a + size_b
            if proposed_size > max_component_size:
                decision = "deferred_component_cap"
            else:
                kept_root, merged_root = sorted((root_a, root_b))
                parent[merged_root] = kept_root
                sizes[kept_root] = proposed_size
                decision = "accepted_merge"
        record = {
            "account_a": account_a,
            "account_b": account_b,
            "relation_score": float(getattr(row, score_column)),
            "component_a_size_before": size_a,
            "component_b_size_before": size_b,
            "proposed_component_size": proposed_size,
            "guardrail_decision": decision,
        }
        if "split" in active.columns:
            record["split"] = str(row.split)
        decisions.append(record)

    components: dict[str, list[str]] = {}
    for account in account_ids:
        components.setdefault(find(account), []).append(account)
    ordered_components = sorted(
        (sorted(members) for members in components.values()), key=lambda x: x[0]
    )
    assignment_records: list[dict[str, str]] = []
    for position, component in enumerate(ordered_components, start=1):
        group_id = f"PRED-{position:05d}"
        assignment_records.extend(
            {"account_id": account, "predicted_group_id": group_id} for account in component
        )
    assignments = pd.DataFrame.from_records(assignment_records).sort_values(
        "account_id", ignore_index=True
    )
    audit_columns = [
        "account_a",
        "account_b",
        "relation_score",
        "component_a_size_before",
        "component_b_size_before",
        "proposed_component_size",
        "guardrail_decision",
    ]
    if "split" in active.columns:
        audit_columns.append("split")
    audit = pd.DataFrame.from_records(decisions, columns=audit_columns)
    return assignments, audit


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
    prediction_column: str = "enhanced_match",
) -> pd.DataFrame:
    """Summarize evidence without claiming legal, familial, or fraud relationships."""
    member_data = assignments.merge(
        signatures[["account_id", "family_token_norm"]], on="account_id", how="left"
    )
    rows: list[dict[str, object]] = []
    positive_pairs = predicted_pairs[predicted_pairs[prediction_column] == 1]
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
