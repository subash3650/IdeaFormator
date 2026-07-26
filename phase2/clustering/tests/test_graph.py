"""Tests for RelationshipGraph and RelationshipEdge."""

from __future__ import annotations

from phase2.clustering.graph import RelationshipEdge, RelationshipGraph


def _make_edge(source: str, target: str, similarity: float = 0.9) -> RelationshipEdge:
    return RelationshipEdge(
        source_id=source,
        target_id=target,
        similarity=similarity,
        confidence=similarity * 0.9,
    )


class TestRelationshipEdge:
    def test_frozen(self) -> None:
        edge = _make_edge("a", "b", 0.95)
        assert edge.source_id == "a"
        assert edge.target_id == "b"
        assert edge.similarity == 0.95

    def test_sorted_pair(self) -> None:
        edge = _make_edge("z", "a")
        assert edge.sorted_pair == ("a", "z")


class TestRelationshipGraph:
    def test_add_edge(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b", 0.9))
        assert "a" in g
        assert "b" in g
        assert len(g) == 2

    def test_nodes(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b"))
        g.add_edge(_make_edge("b", "c"))
        assert g.nodes() == {"a", "b", "c"}

    def test_edges(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b"))
        g.add_edge(_make_edge("b", "c"))
        assert len(g.edges()) == 2

    def test_neighbors(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b", 0.9))
        g.add_edge(_make_edge("a", "c", 0.8))
        nbrs = g.neighbors("a")
        assert nbrs == {"b": 0.9, "c": 0.8}

    def test_degree(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b"))
        g.add_edge(_make_edge("a", "c"))
        g.add_edge(_make_edge("a", "d"))
        assert g.degree("a") == 3
        assert g.degree("b") == 1

    def test_weighted_degree(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b", 0.9))
        g.add_edge(_make_edge("a", "c", 0.7))
        assert abs(g.weighted_degree("a") - 1.6) < 1e-9

    def test_edge_weight(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b", 0.85))
        assert g.edge_weight("a", "b") == 0.85
        assert g.edge_weight("b", "a") == 0.85
        assert g.edge_weight("a", "c") is None

    def test_has_edge(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b"))
        assert g.has_edge("a", "b")
        assert g.has_edge("b", "a")
        assert not g.has_edge("a", "c")

    def test_get_edge(self) -> None:
        g = RelationshipGraph()
        e = _make_edge("a", "b", 0.95)
        g.add_edge(e)
        retrieved = g.get_edge("a", "b")
        assert retrieved is not None
        assert retrieved.similarity == 0.95

    def test_subgraph(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b"))
        g.add_edge(_make_edge("b", "c"))
        g.add_edge(_make_edge("c", "d"))
        sub = g.subgraph({"a", "b"})
        assert sub.nodes() == {"a", "b"}
        assert len(sub.edges()) == 1

    def test_from_edges(self) -> None:
        edges = [_make_edge("a", "b"), _make_edge("c", "d")]
        g = RelationshipGraph.from_edges(edges)
        assert len(g) == 4
        assert len(g.edges()) == 2

    def test_connected_components(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b"))
        g.add_edge(_make_edge("b", "c"))
        g.add_edge(_make_edge("d", "e"))
        components = g.connected_components()
        assert len(components) == 2
        component_sets = [frozenset(c) for c in components]
        assert frozenset({"a", "b", "c"}) in component_sets
        assert frozenset({"d", "e"}) in component_sets

    def test_filter_edges(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b", 0.9))
        g.add_edge(_make_edge("c", "d", 0.5))
        filtered = g.filter_edges(0.8)
        assert len(filtered.edges()) == 1
        assert filtered.has_edge("a", "b")

    def test_size(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "b"))
        g.add_edge(_make_edge("b", "c"))
        nodes, edges = g.size()
        assert nodes == 3
        assert edges == 2

    def test_add_edges(self) -> None:
        g = RelationshipGraph()
        edges = [_make_edge("a", "b"), _make_edge("c", "d"), _make_edge("e", "f")]
        g.add_edges(edges)
        assert len(g) == 6
        assert len(g.edges()) == 3

    def test_empty_graph(self) -> None:
        g = RelationshipGraph()
        assert len(g) == 0
        assert g.edges() == []
        assert g.nodes() == set()
        assert g.connected_components() == []

    def test_single_node(self) -> None:
        g = RelationshipGraph()
        g.add_edge(_make_edge("a", "a", 0.5))
        assert "a" in g
