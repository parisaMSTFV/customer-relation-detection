"""Reproducible evaluation and review visuals."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

COLORS = {
    "navy": "#17324D",
    "teal": "#4A9D8F",
    "gray": "#78808E",
    "amber": "#D69E3D",
    "red": "#C85C5C",
    "ivory": "#F7F3EA",
}


def plot_model_comparison(
    summary: dict[str, dict[str, float | int]],
    cluster_scores: dict[str, float],
    output_path: Path,
) -> None:
    """Compare pair classification and connected-component quality."""
    metrics = ["precision", "recall", "f1"]
    baseline = [float(summary["exact_key_baseline"][metric]) for metric in metrics]
    enhanced = [float(summary["scored_resolution"][metric]) for metric in metrics]
    guarded = [float(summary["guarded_resolution"][metric]) for metric in metrics]
    baseline.append(cluster_scores["exact_key_baseline"])
    enhanced.append(cluster_scores["scored_resolution"])
    guarded.append(cluster_scores["guarded_resolution"])
    labels = ["Precision", "Recall", "F1", "Cluster ARI"]
    positions = np.arange(len(labels))
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    baseline_bars = axis.bar(
        positions - 0.24, baseline, width=0.24, color=COLORS["gray"], label="Exact key"
    )
    enhanced_bars = axis.bar(
        positions,
        enhanced,
        width=0.24,
        color=COLORS["teal"],
        label="Scored links",
    )
    guarded_bars = axis.bar(
        positions + 0.24,
        guarded,
        width=0.24,
        color=COLORS["navy"],
        label="Size-guarded groups",
    )
    axis.set_xticks(positions, labels)
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Score")
    axis.set_title("Held-out building evaluation", loc="left", weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    axis.bar_label(baseline_bars, fmt="%.2f", padding=2)
    axis.bar_label(enhanced_bars, fmt="%.2f", padding=2)
    axis.bar_label(guarded_bars, fmt="%.2f", padding=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_threshold_curve(
    curve: pd.DataFrame,
    selected_threshold: float,
    output_path: Path,
) -> None:
    """Plot validation-only component policy metrics."""
    fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    for metric, color in (
        ("precision", COLORS["navy"]),
        ("recall", COLORS["amber"]),
        ("f1", COLORS["teal"]),
    ):
        axis.plot(curve["threshold"], curve[metric], label=metric.title(), color=color)
    axis.axvline(
        selected_threshold,
        color=COLORS["red"],
        linestyle="--",
        label=f"Selected {selected_threshold:.2f}",
    )
    axis.set_ylim(0, 1.05)
    axis.set_xlabel("Relation score threshold")
    axis.set_ylabel("Component pair score")
    axis.set_title("Graph-policy selection on validation buildings", loc="left", weight="bold")
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_noise_recall(noise_recall: pd.DataFrame, output_path: Path) -> None:
    """Compare baseline and scored recall across address-noise levels."""
    positions = np.arange(len(noise_recall))
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    baseline = axis.bar(
        positions - 0.18,
        noise_recall["baseline_recall"],
        width=0.36,
        color=COLORS["gray"],
        label="Exact key",
    )
    enhanced = axis.bar(
        positions + 0.18,
        noise_recall["enhanced_recall"],
        width=0.36,
        color=COLORS["teal"],
        label="Scored resolution",
    )
    axis.set_xticks(positions, noise_recall["noise_level"].astype(str).str.title())
    axis.set_ylim(0, 1.08)
    axis.set_ylabel("Positive-pair recall")
    axis.set_title("Recall by maximum pair noise", loc="left", weight="bold")
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="upper right")
    axis.bar_label(baseline, fmt="%.2f", padding=2)
    axis.bar_label(enhanced, fmt="%.2f", padding=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_review_graph(
    test_pairs: pd.DataFrame,
    assignments: pd.DataFrame,
    output_path: Path,
    prediction_column: str = "guardrail_match",
) -> None:
    """Plot one complex synthetic building using operationally accepted links."""
    account_counts = test_pairs.groupby("building_id")[["account_a", "account_b"]].agg(
        lambda column: set(column)
    )
    account_counts["accounts"] = [
        left | right
        for left, right in zip(
            account_counts["account_a"], account_counts["account_b"], strict=True
        )
    ]
    building_id = max(
        account_counts.index, key=lambda value: len(account_counts.loc[value, "accounts"])
    )
    building_pairs = test_pairs[test_pairs["building_id"] == building_id]
    accounts = sorted(account_counts.loc[building_id, "accounts"])
    graph = nx.Graph()
    graph.add_nodes_from(accounts)
    for row in building_pairs[building_pairs[prediction_column] == 1].itertuples(index=False):
        graph.add_edge(row.account_a, row.account_b, score=row.relation_score)
    assignment_map = assignments.set_index("account_id")["predicted_group_id"].to_dict()
    group_codes = {group: index for index, group in enumerate(sorted(set(assignment_map.values())))}
    node_colors = [group_codes[assignment_map[node]] for node in graph.nodes]
    widths = [1 + 4 * graph.edges[edge]["score"] for edge in graph.edges]
    position = nx.spring_layout(graph, seed=42, k=0.9)
    labels = {node: node.replace("ACC-", "A") for node in graph.nodes}
    fig, axis = plt.subplots(figsize=(10, 7), constrained_layout=True)
    fig.patch.set_facecolor(COLORS["ivory"])
    axis.set_facecolor("#FCFAF5")
    nx.draw_networkx_edges(graph, position, width=widths, alpha=0.45, ax=axis)
    nx.draw_networkx_nodes(
        graph,
        position,
        node_color=node_colors,
        cmap="tab10",
        node_size=650,
        edgecolors="white",
        linewidths=1,
        ax=axis,
    )
    nx.draw_networkx_labels(graph, position, labels=labels, font_size=8, ax=axis)
    axis.set_title(f"Synthetic review graph for building {building_id}", loc="left", weight="bold")
    axis.text(
        0,
        -0.03,
        "Color = guarded component · edge width = accepted-link score · no relationship type is asserted",
        transform=axis.transAxes,
        color=COLORS["gray"],
    )
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
