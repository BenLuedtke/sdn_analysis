"""
Unit tests for the 50% rule ownership graph traversal.

Graphs are constructed explicitly with known correct answers so the traversal
can be verified without external data. Covers:
  - Direct 50% ownership
  - Indirect chain ownership
  - Aggregate ownership from multiple SDN owners
  - Cycle handling
  - Missing ownership percentages
  - Edge cases (sub-threshold, isolated nodes)
"""

import networkx as nx
import pytest

from sanctions.graph import (
    compute_sdn_fraction,
    flag_blocked_entities,
    ownership_paths,
    path_weight,
)


def make_graph(edges: list[tuple]) -> nx.DiGraph:
    """Build a DiGraph from (owner, asset, weight) triples."""
    G = nx.DiGraph()
    for owner, asset, weight in edges:
        G.add_edge(owner, asset, weight=weight)
    return G


# ── compute_sdn_fraction ──────────────────────────────────────────────────────

class TestComputeSdnFraction:
    def test_sdn_node_returns_one(self):
        G = make_graph([("SDN_A", "CO_1", 0.6)])
        assert compute_sdn_fraction(G, "SDN_A", frozenset({"SDN_A"})) == 1.0

    def test_direct_majority_ownership(self):
        # SDN_A owns 60% of CO_1 → CO_1 should have 0.60 SDN fraction
        G = make_graph([("SDN_A", "CO_1", 0.6)])
        assert compute_sdn_fraction(G, "CO_1", frozenset({"SDN_A"})) == pytest.approx(0.6)

    def test_direct_minority_ownership(self):
        # SDN_A owns 30% of CO_1 → fraction = 0.30
        G = make_graph([("SDN_A", "CO_1", 0.3)])
        assert compute_sdn_fraction(G, "CO_1", frozenset({"SDN_A"})) == pytest.approx(0.3)

    def test_indirect_chain(self):
        # SDN_A → CO_1 (60%) → CO_2 (70%)
        # SDN fraction of CO_2 = 0.60 × 0.70 = 0.42
        G = make_graph([("SDN_A", "CO_1", 0.6), ("CO_1", "CO_2", 0.7)])
        assert compute_sdn_fraction(G, "CO_2", frozenset({"SDN_A"})) == pytest.approx(0.42)

    def test_three_hop_chain(self):
        # 0.8 × 0.8 × 0.8 = 0.512
        G = make_graph([
            ("SDN_A", "CO_1", 0.8),
            ("CO_1",  "CO_2", 0.8),
            ("CO_2",  "CO_3", 0.8),
        ])
        assert compute_sdn_fraction(G, "CO_3", frozenset({"SDN_A"})) == pytest.approx(0.512)

    def test_aggregate_two_sdn_owners(self):
        # SDN_A owns 30% and SDN_B owns 30% → aggregate = 0.60 → blocked
        G = make_graph([("SDN_A", "CO_1", 0.3), ("SDN_B", "CO_1", 0.3)])
        frac = compute_sdn_fraction(G, "CO_1", frozenset({"SDN_A", "SDN_B"}))
        assert frac == pytest.approx(0.60)

    def test_caps_at_one(self):
        # Two SDNs each owning 80% → aggregate would be 1.60, must cap at 1.0
        G = make_graph([("SDN_A", "CO_1", 0.8), ("SDN_B", "CO_1", 0.8)])
        frac = compute_sdn_fraction(G, "CO_1", frozenset({"SDN_A", "SDN_B"}))
        assert frac == pytest.approx(1.0)

    def test_no_sdn_connection(self):
        G = make_graph([("CLEAN_A", "CO_1", 0.9)])
        frac = compute_sdn_fraction(G, "CO_1", frozenset({"SDN_X"}))
        assert frac == pytest.approx(0.0)

    def test_cycle_handling(self):
        # A → B → C → A (cycle); SDN_A → A (60%)
        G = nx.DiGraph()
        G.add_edge("SDN_A", "A", weight=0.6)
        G.add_edge("A", "B", weight=0.5)
        G.add_edge("B", "C", weight=0.5)
        G.add_edge("C", "A", weight=0.5)   # creates cycle
        # Should not infinite-loop; SDN fraction of B = 0.6 × 0.5 = 0.30
        frac = compute_sdn_fraction(G, "B", frozenset({"SDN_A"}))
        assert 0.0 <= frac <= 1.0  # just verify it terminates with a valid value

    def test_missing_weight_uses_unknown(self):
        # Edge without weight; unknown_weight=1.0 → treat as full ownership
        G = nx.DiGraph()
        G.add_edge("SDN_A", "CO_1")  # no weight attribute
        frac = compute_sdn_fraction(G, "CO_1", frozenset({"SDN_A"}), unknown_weight=1.0)
        assert frac == pytest.approx(1.0)

    def test_missing_weight_conservative_half(self):
        G = nx.DiGraph()
        G.add_edge("SDN_A", "CO_1")
        frac = compute_sdn_fraction(G, "CO_1", frozenset({"SDN_A"}), unknown_weight=0.5)
        assert frac == pytest.approx(0.5)

    def test_isolated_non_sdn_node(self):
        G = nx.DiGraph()
        G.add_node("CO_ISOLATED")
        frac = compute_sdn_fraction(G, "CO_ISOLATED", frozenset({"SDN_A"}))
        assert frac == pytest.approx(0.0)


# ── flag_blocked_entities ─────────────────────────────────────────────────────

class TestFlagBlockedEntities:
    def test_majority_owned_flagged(self):
        G = make_graph([("SDN_A", "CO_1", 0.6)])
        blocked = flag_blocked_entities(G, {"SDN_A"})
        assert "CO_1" in blocked
        assert "SDN_A" not in blocked  # SDN itself never in blocked dict

    def test_minority_owned_not_flagged(self):
        G = make_graph([("SDN_A", "CO_1", 0.49)])
        blocked = flag_blocked_entities(G, {"SDN_A"})
        assert "CO_1" not in blocked

    def test_exactly_threshold_flagged(self):
        G = make_graph([("SDN_A", "CO_1", 0.50)])
        blocked = flag_blocked_entities(G, {"SDN_A"})
        assert "CO_1" in blocked

    def test_aggregate_aggregate_triggers(self):
        # Two SDNs each at 30% → sum 60% → blocked
        G = make_graph([("SDN_A", "CO_1", 0.30), ("SDN_B", "CO_1", 0.30)])
        blocked = flag_blocked_entities(G, {"SDN_A", "SDN_B"})
        assert "CO_1" in blocked

    def test_cascade_block(self):
        # SDN → A (80%) → B (80%). Effective ownership of B = 0.64 → blocked.
        # SDN → A (60%) → B (60%) = 0.36 would NOT be blocked (below 50%).
        G = make_graph([("SDN_A", "A", 0.8), ("A", "B", 0.8)])
        blocked = flag_blocked_entities(G, {"SDN_A"})
        assert "A" in blocked
        assert "B" in blocked
        assert blocked["B"] == pytest.approx(0.64)

    def test_cascade_does_not_over_block(self):
        # SDN → A (60%) → B (60%) = 0.36 — should NOT be blocked
        G = make_graph([("SDN_A", "A", 0.6), ("A", "B", 0.6)])
        blocked = flag_blocked_entities(G, {"SDN_A"})
        assert "A" in blocked   # 60% — blocked
        assert "B" not in blocked  # 36% — not blocked

    def test_custom_threshold(self):
        G = make_graph([("SDN_A", "CO_1", 0.40)])
        blocked_50 = flag_blocked_entities(G, {"SDN_A"}, threshold=0.50)
        blocked_30 = flag_blocked_entities(G, {"SDN_A"}, threshold=0.30)
        assert "CO_1" not in blocked_50
        assert "CO_1" in blocked_30

    def test_returns_fraction_values(self):
        G = make_graph([("SDN_A", "CO_1", 0.75)])
        blocked = flag_blocked_entities(G, {"SDN_A"})
        assert blocked["CO_1"] == pytest.approx(0.75)


# ── path utilities ────────────────────────────────────────────────────────────

class TestPathUtils:
    def test_ownership_paths_found(self):
        G = make_graph([("SDN_A", "CO_1", 0.6), ("CO_1", "CO_2", 0.7)])
        paths = ownership_paths(G, "SDN_A", "CO_2")
        assert len(paths) == 1
        assert paths[0] == ["SDN_A", "CO_1", "CO_2"]

    def test_path_weight_product(self):
        G = make_graph([("SDN_A", "CO_1", 0.6), ("CO_1", "CO_2", 0.7)])
        w = path_weight(G, ["SDN_A", "CO_1", "CO_2"])
        assert w == pytest.approx(0.42)

    def test_no_path_returns_empty(self):
        G = make_graph([("SDN_A", "CO_1", 0.6)])
        paths = ownership_paths(G, "SDN_A", "CO_UNCONNECTED")
        assert paths == []
