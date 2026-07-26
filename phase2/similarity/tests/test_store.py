"""Tests for SemanticRelationshipStore."""

from __future__ import annotations

from pathlib import Path

from phase2.embeddings.schema import SourceType
from phase2.similarity.schema import SemanticRelationship
from phase2.similarity.store import SemanticRelationshipStore


def _make_rel(
    source_id: str = "src1",
    target_id: str = "tgt1",
    source_type: SourceType = SourceType.observation,
    target_type: SourceType = SourceType.observation,
) -> SemanticRelationship:
    return SemanticRelationship(
        relationship_id=f"rel_{source_id}_{target_id}",
        source_type=source_type,
        source_id=source_id,
        target_type=target_type,
        target_id=target_id,
        similarity_score=0.9,
        confidence=0.8,
        metric="cosine",
        provider="cosine",
        model_fingerprint="fp",
        version="1.0",
    )


class TestSemanticRelationshipStore:
    def test_save_and_load(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        rels = [_make_rel(), _make_rel(source_id="s2", target_id="t2")]
        store.save(rels)
        loaded = store.load()
        assert len(loaded) == 2

    def test_save_empty(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([])
        # Empty parquet file is still written with schema
        assert store.exists()
        loaded = store.load()
        assert len(loaded) == 0

    def test_overwrites_on_save(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([_make_rel()])
        store.save([_make_rel(source_id="new")])
        loaded = store.load()
        assert len(loaded) == 1
        assert loaded[0].source_id == "new"

    def test_append(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([_make_rel()])
        store.append([_make_rel(source_id="appended")])
        loaded = store.load()
        assert len(loaded) == 2

    def test_count(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        assert store.count() == 0
        store.save([_make_rel(), _make_rel(source_id="s2")])
        assert store.count() == 2

    def test_exists(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        assert not store.exists()
        store.save([_make_rel()])
        assert store.exists()

    def test_load_df(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        store.save([_make_rel()])
        df = store.load_df()
        assert df.height == 1
        assert "source_id" in df.columns

    def test_roundtrip_metadata(self, tmp_path: Path) -> None:
        store = SemanticRelationshipStore(tmp_path)
        rel = _make_rel()
        # Rebuild with metadata
        rel_with_meta = SemanticRelationship(
            relationship_id=rel.relationship_id,
            source_type=rel.source_type,
            source_id=rel.source_id,
            target_type=rel.target_type,
            target_id=rel.target_id,
            similarity_score=rel.similarity_score,
            confidence=rel.confidence,
            metric=rel.metric,
            provider=rel.provider,
            model_fingerprint=rel.model_fingerprint,
            version=rel.version,
            shared_entities=["e1", "e2"],
            shared_categories=["c1"],
            metadata={"key": "value"},
        )
        store.save([rel_with_meta])
        loaded = store.load()
        assert loaded[0].shared_entities == ["e1", "e2"]
        assert loaded[0].metadata == {"key": "value"}
