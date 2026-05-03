"""
OFAC 50% Rule ownership graph traversal.

The 50% rule (OFAC FAQ 398): an entity is considered blocked if one or more
SDN entities own, directly or indirectly in aggregate, 50% or more of that
entity's equity interests. "Aggregate" means the contributions of all SDN
owners are summed.

Algorithm:
    For each non-SDN target node, compute the aggregate fraction of that node's
    equity attributable to SDN entities by recursively traversing the ownership
    graph upstream. Ownership fractions multiply along each path (chain rule);
    contributions from distinct SDN ancestors sum.

Cycle handling:
    Ownership graphs can contain cycles (joint ventures, cross-holdings). The
    visited-node set breaks cycles by refusing to re-enter a node within one
    traversal, yielding a conservative (lower-bound) estimate of SDN ownership.

Missing ownership percentages:
    When an ownership edge lacks a percentage, the `unknown_weight` parameter
    controls the assumed value. OFAC guidance is that unknown percentages should
    not be treated as zero — 1.0 (assume full control) is the conservative
    compliance posture; 0.5 is a middle ground. The default here is 1.0.

Graph convention:
    Directed edges point from owner to owned: owner → asset.
    Edge attribute `weight` holds the ownership fraction in [0, 1].
"""

from __future__ import annotations

from typing import Optional

import networkx as nx


def compute_sdn_fraction(
    G: nx.DiGraph,
    node: str,
    sdn_nodes: frozenset[str],
    visited: frozenset[str] = frozenset(),
    unknown_weight: float = 1.0,
) -> float:
    """
    Compute the aggregate SDN-ownership fraction for a single node.

    Returns a float in [0, 1]. A value ≥ 0.5 triggers the 50% rule.

    Args:
        G:             Directed ownership graph (owner → asset edges).
        node:          Node to evaluate.
        sdn_nodes:     Set of explicitly sanctioned entity IDs.
        visited:       Nodes already visited in this traversal (cycle guard).
        unknown_weight: Assumed ownership fraction when edge weight is missing.
    """
    if node in visited:
        return 0.0   # cycle — stop recursing

    if node in sdn_nodes:
        return 1.0   # base case: node is itself sanctioned

    visited = visited | {node}
    total = 0.0

    for owner in G.predecessors(node):
        weight = G[owner][node].get("weight", unknown_weight)
        owner_sdn_fraction = compute_sdn_fraction(
            G, owner, sdn_nodes, visited, unknown_weight
        )
        total += weight * owner_sdn_fraction

    return min(total, 1.0)  # cap at 1.0; cannot be "more than fully owned"


def flag_blocked_entities(
    G: nx.DiGraph,
    sdn_nodes: frozenset[str] | set[str],
    threshold: float = 0.50,
    unknown_weight: float = 1.0,
) -> dict[str, float]:
    """
    Apply the 50% rule to every non-SDN node in the graph.

    Returns a dict mapping entity_id → sdn_fraction for all entities where
    sdn_fraction >= threshold (i.e., effectively blocked but not explicitly listed).

    Args:
        G:             Directed ownership graph.
        sdn_nodes:     Set of explicitly sanctioned entity IDs.
        threshold:     Ownership fraction that triggers blocking (default 0.50).
        unknown_weight: Assumed weight for edges without a percentage.
    """
    sdn_frozen = frozenset(sdn_nodes)
    blocked = {}

    for node in G.nodes():
        if node in sdn_frozen:
            continue
        fraction = compute_sdn_fraction(G, node, sdn_frozen, unknown_weight=unknown_weight)
        if fraction >= threshold:
            blocked[node] = round(fraction, 4)

    return blocked


def ownership_paths(
    G: nx.DiGraph,
    source: str,
    target: str,
    cutoff: int = 6,
) -> list[list[str]]:
    """
    Find all directed paths from source (owner) to target (asset) up to cutoff hops.
    Useful for explaining why an entity was flagged.
    """
    try:
        return list(nx.all_simple_paths(G, source, target, cutoff=cutoff))
    except (nx.NodeNotFound, nx.NetworkXError):
        return []


def path_weight(G: nx.DiGraph, path: list[str], unknown_weight: float = 1.0) -> float:
    """Return the product of edge weights along a path (effective ownership %)."""
    weight = 1.0
    for i in range(len(path) - 1):
        weight *= G[path[i]][path[i + 1]].get("weight", unknown_weight)
    return weight


def explain_blocking(
    G: nx.DiGraph,
    node: str,
    sdn_nodes: frozenset[str] | set[str],
    unknown_weight: float = 1.0,
    cutoff: int = 6,
) -> list[dict]:
    """
    Return a human-readable explanation of why a node is (or isn't) blocked.

    Returns a list of dicts, one per SDN ancestor that contributes ownership:
        {sdn_id, path, path_weight, sdn_contribution}
    """
    sdn_frozen = frozenset(sdn_nodes)
    sdn_ancestors = [n for n in nx.ancestors(G, node) if n in sdn_frozen]

    explanations = []
    for sdn in sdn_ancestors:
        paths = ownership_paths(G, sdn, node, cutoff=cutoff)
        if not paths:
            continue
        # Take the path with the highest effective weight
        best_path = max(paths, key=lambda p: path_weight(G, p, unknown_weight))
        pw = path_weight(G, best_path, unknown_weight)
        explanations.append({
            "sdn_id":          sdn,
            "path":            best_path,
            "path_weight":     round(pw, 4),
            "sdn_contribution": round(pw, 4),
        })

    explanations.sort(key=lambda x: x["sdn_contribution"], reverse=True)
    return explanations
