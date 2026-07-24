"""Tests for RelationshipSearcher."""

from __future__ import annotations

from pathlib import Path

from phase2.embeddings.schema import SourceType
from phase2.similarity.schema import SemanticRelationship
from phase2.similarity.search import RelationshipSearcher
from phase2.similarity.store import SemanticRelationshipStore


def _make_rel(
    source_id: str = "s1",
    target_id: str = "t1",
    source_type: SourceType = SourceType.observation,
    target_type: SourceType = SourceType.observation,
    similarity: float = 0.9,
) -> SemanticRelationship:
    return SemanticRelationship(
        relationship_id=f"rel_{source_id}_{target_id}",
        source_type=source_type,
        source_id=source_id,
        target_type=target_type,
        target_id=target_id,
        similarity_score=similarity,
        confidence=0.8,
        metric="cosine",
        provider="cosine",
        model_fingerprint="fp",
        version="1.0",
    )


class TestRelationshipSearcher:
    def test_search_by_source(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([
            _make_rel(source_id="s1", target_id="t1", similarity=0.9),
            _make_rel(source_id="s1", target_id="t2", similarity=0.8),
            _make_rel(source_id="s2", target_id="t3", similarity=0.7),
        ])
        searcher = RelationshipSearcher(store)
        results = searcher.search_by_source("s1")
        assert len(results) == 2
        # Should be sorted by similarity descending
        assert results[0].similarity_score >= results[1].similarity_score

    def test_search_by_target(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([
            _make_rel(source_id="s1", target_id="t1"),
            _make_rel(source_id="s2", target_id="t1"),
        ])
        searcher = RelationshipSearcher(store)
        results = searcher.search_by_target("t1")
        assert len(results) == 2

    def test_search_by_pair(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([
            _make_rel(source_id="s1", target_id="t1", similarity=0.9),
            _make_rel(source_id="s1", target_id="t2", similarity=0.8),
        ])
        searcher = RelationshipSearcher(store)
        result = searcher.search_by_pair("s1", "t1")
        assert result is not None
        assert result.similarity_score == 0.9

    def test_search_by_pair_reverse(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([_make_rel(source_id="s1", target_id="t1")])
        searcher = RelationshipSearcher(store)
        result = searcher.search_by_pair("t1", "s1")
        assert result is not None

    def test_search_by_pair_not_found(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        searcher = RelationshipSearcher(store)
        assert searcher.search_by_pair("x", "y") is None

    def test_get_related_ids(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([
            _make_rel(source_id="s1", target_id="t1"),
            _make_rel(source_id="s1", target_id="t2"),
            _make_rel(source_id="t1", target_id="s1"),  # reverse
        ])
        searcher = RelationshipSearcher(store)
        ids = searcher.get_related_ids("s1")
        assert "t1" in ids
        assert "t2" in ids

    def test_empty_store(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        searcher = RelationshipSearcher(store)
        assert searcher.search_by_source("x") == []
        assert searcher.search_by_target("x") == []
        assert searcher.get_related_ids("x") == []
