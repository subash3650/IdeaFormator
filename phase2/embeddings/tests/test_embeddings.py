"""Tests for the embedding infrastructure."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from phase2.embeddings.cache import EmbeddingCache
from phase2.embeddings.config import EmbeddingEngineConfig, load_embedding_config
from phase2.embeddings.engine import EmbeddingEngine, _make_embedding_id
from phase2.embeddings.exporter import generate_quality_report, write_manifest
from phase2.embeddings.metrics import compute_stats
from phase2.embeddings.schema import (
    EmbeddingJob,
    EmbeddingManifest,
    EmbeddingRecord,
    SearchResult,
    SourceType,
)
from phase2.embeddings.search import LinearIndex, build_index
from phase2.embeddings.store import EmbeddingStore


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_embedding_record_frozen(self) -> None:
        rec = EmbeddingRecord(
            embedding_id="abc123",
            source_id="src1",
            source_type=SourceType.observation,
            provider="test",
            model="m",
            dimension=3,
            embedding=[0.1, 0.2, 0.3],
            created_at="2025-01-01T00:00:00",
        )
        assert rec.embedding_id == "abc123"
        assert rec.to_vector().shape == (3,)

    def test_search_result_frozen(self) -> None:
        r = SearchResult(
            embedding_id="e1",
            source_id="s1",
            source_type=SourceType.evidence,
            provider="p",
            model="m",
            similarity=0.95,
        )
        assert r.similarity == 0.95

    def test_embedding_job(self) -> None:
        job = EmbeddingJob(
            source_type=SourceType.custom,
            source_path=Path("/a/b.parquet"),
            output_path=Path("/out"),
        )
        assert job.source_type == SourceType.custom

    def test_manifest_serialization(self) -> None:
        m = EmbeddingManifest(
            provider="test",
            model="m",
            model_version="v1",
            dimension=3,
            normalize=True,
            num_vectors=10,
            sources={},
            created_at="now",
        )
        d = m.model_dump(mode="json")
        assert d["provider"] == "test"
        assert d["num_vectors"] == 10


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestConfig:
    def test_default_config(self) -> None:
        cfg = EmbeddingEngineConfig()
        assert cfg.provider == "sentence_transformers"
        assert cfg.model == "all-MiniLM-L6-v2"
        assert cfg.dimension == 384
        assert cfg.normalize is True

    def test_load_embedding_config_missing_file(self) -> None:
        cfg = load_embedding_config("nonexistent.yaml")
        assert isinstance(cfg, EmbeddingEngineConfig)

    def test_load_embedding_config_from_yaml(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("embedding:\n  provider: ollama\n  model: llama3\n  dimension: 4096\n")
            path = f.name
        cfg = load_embedding_config(path)
        assert cfg.provider == "ollama"
        assert cfg.model == "llama3"
        assert cfg.dimension == 4096
        Path(path).unlink()

    def test_source_paths_default(self) -> None:
        cfg = EmbeddingEngineConfig()
        assert SourceType.observation in cfg.source_paths


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class TestCache:
    def test_add_and_contains(self) -> None:
        cache = EmbeddingCache()
        cache.add("abc")
        assert cache.contains("abc")
        assert not cache.contains("def")

    def test_remove(self) -> None:
        cache = EmbeddingCache()
        cache.add("abc")
        cache.remove("abc")
        assert not cache.contains("abc")

    def test_clear(self) -> None:
        cache = EmbeddingCache()
        cache.add("a")
        cache.add("b")
        cache.clear()
        assert cache.size == 0

    def test_lru_eviction(self) -> None:
        cache = EmbeddingCache(maxsize=2)
        cache.add("a")
        cache.add("b")
        cache.add("c")  # evicts "a"
        assert not cache.contains("a")
        assert cache.contains("b")
        assert cache.contains("c")
        assert cache.size == 2


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class TestStore:
    @pytest.fixture
    def tmp_dir(self) -> Path:
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    def _sample_records(self, n: int = 3) -> list[EmbeddingRecord]:
        return [
            EmbeddingRecord(
                embedding_id=f"e{i}",
                source_id=f"s{i}",
                source_type=SourceType.observation,
                provider="test",
                model="m",
                dimension=2,
                embedding=[float(i), float(i + 1)],
                created_at="2025-01-01T00:00:00",
            )
            for i in range(n)
        ]

    def test_write_and_read(self, tmp_dir: Path) -> None:
        store = EmbeddingStore(tmp_dir)
        records = self._sample_records(2)
        store.write(records, "observation")
        df = store.read("observation")
        assert df.height == 2
        assert "embedding_id" in df.columns

    def test_append(self, tmp_dir: Path) -> None:
        store = EmbeddingStore(tmp_dir)
        store.write(self._sample_records(2), "observation")
        store.append(self._sample_records(1), "observation")
        df = store.read("observation")
        assert df.height == 3

    def test_read_all(self, tmp_dir: Path) -> None:
        store = EmbeddingStore(tmp_dir)
        store.write(self._sample_records(2), "observation")
        store.write(self._sample_records(1), "evidence")
        df = store.read_all()
        assert df.height == 3

    def test_exists(self, tmp_dir: Path) -> None:
        store = EmbeddingStore(tmp_dir)
        assert not store.exists("observation")
        store.write(self._sample_records(1), "observation")
        assert store.exists("observation")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    def _sample_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "embedding_id": ["e1", "e2", "e3"],
            "source_id": ["s1", "s2", "s3"],
            "source_type": ["observation", "evidence", "problem_signal"],
            "provider": ["p"] * 3,
            "model": ["m"] * 3,
            "model_version": ["v1"] * 3,
            "dimension": [2] * 3,
            "embedding": [[1.0, 0.0], [0.0, 1.0], [0.707, 0.707]],
            "text_snippet": ["a", "b", "c"],
            "created_at": ["now"] * 3,
        })

    def test_linear_index(self) -> None:
        df = self._sample_df()
        index = build_index(df)
        assert index.size == 3
        assert index.dimension == 2

    def test_search_returns_results(self) -> None:
        df = self._sample_df()
        index = build_index(df)
        query = np.array([1.0, 0.0])
        results = index.search(query, k=2)
        assert len(results) == 2
        assert results[0].similarity >= results[1].similarity

    def test_search_normalizes_query(self) -> None:
        df = self._sample_df()
        index = build_index(df)
        query = np.array([2.0, 0.0])
        results = index.search(query, k=1)
        assert results[0].source_id == "s1"

    def test_empty_index_raises_on_dimension_mismatch(self) -> None:
        with pytest.raises(ValueError, match="count mismatch"):
            LinearIndex(np.empty((0, 2)), pl.DataFrame({"a": [1]}))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_compute_stats_empty(self) -> None:
        df = pl.DataFrame({"embedding": [], "source_type": []})
        stats = compute_stats(df)
        assert stats.total_vectors == 0

    def test_compute_stats_with_data(self) -> None:
        df = pl.DataFrame({
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "source_type": ["a", "b"],
            "text_snippet": ["hello", None],
        })
        stats = compute_stats(df)
        assert stats.total_vectors == 2
        assert stats.dimension == 2
        assert stats.null_text_snippets == 1
        assert abs(stats.mean_norm - 1.0) < 0.01


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------


class TestExporter:
    def test_write_manifest(self, tmp_path: Path) -> None:
        m = EmbeddingManifest(
            provider="p",
            model="m",
            model_version="v1",
            dimension=3,
            normalize=True,
            num_vectors=5,
            sources={},
            created_at="now",
        )
        path = write_manifest(m, tmp_path)
        assert path.exists()
        with open(path) as f:
            data = json.load(f)
        assert data["provider"] == "p"
        assert data["num_vectors"] == 5

    def test_quality_report(self, tmp_path: Path) -> None:
        df = pl.DataFrame({
            "embedding": [[1.0, 0.0], [0.0, 1.0]],
            "source_type": ["obs", "ev"],
            "text_snippet": ["a", "b"],
        })
        path = generate_quality_report(df, tmp_path)
        assert path.exists()
        text = path.read_text()
        assert "PASS" in text


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TestEngine:
    def test_make_embedding_id_deterministic(self) -> None:
        a = _make_embedding_id("s1", "p", "m", "v1")
        b = _make_embedding_id("s1", "p", "m", "v1")
        assert a == b
        c = _make_embedding_id("s2", "p", "m", "v1")
        assert a != c

    def test_generate_with_missing_source(self) -> None:
        cfg = EmbeddingEngineConfig(
            source_paths={SourceType.custom: Path("/nonexistent/file.parquet")},
        )
        engine = EmbeddingEngine(cfg)
        result = engine.generate()
        assert result["total_input"] == 0

    def test_verify_empty(self, tmp_path: Path) -> None:
        cfg = EmbeddingEngineConfig(
            source_paths={SourceType.observation: Path("/nonexistent.parquet")},
            output_dir=tmp_path / "empty",
        )
        engine = EmbeddingEngine(cfg)
        v = engine.verify()
        assert v["valid"] is False

    def test_generate_from_parquet(self, tmp_path: Path) -> None:
        source = tmp_path / "test_source.parquet"
        df = pl.DataFrame({
            "id": ["r1", "r2"],
            "text": ["hello world", "foo bar baz"],
        })
        df.write_parquet(str(source))

        cfg = EmbeddingEngineConfig(
            provider="sentence_transformers",
            model="all-MiniLM-L6-v2",
            source_paths={SourceType.observation: source},
            output_dir=tmp_path / "out",
            store_text=True,
        )
        engine = EmbeddingEngine(cfg)
        result = engine.generate()
        assert result["total_embedded"] > 0
        assert result["total_errors"] == 0

        assert engine.store.exists("observation")
        stats = engine.stats()
        assert stats.total_vectors == 2
        assert stats.dimension == 384

        v = engine.verify()
        assert v["valid"] is True

    def test_search_after_generate(self, tmp_path: Path) -> None:
        source = tmp_path / "search_source.parquet"
        df = pl.DataFrame({
            "id": ["s1"],
            "text": ["machine learning transformer models"],
        })
        df.write_parquet(str(source))

        cfg = EmbeddingEngineConfig(
            source_paths={SourceType.observation: source},
            output_dir=tmp_path / "out2",
            store_text=True,
        )
        engine = EmbeddingEngine(cfg)
        engine.generate()

        results = engine.search("machine learning", k=1)
        assert len(results) == 1
        assert results[0].similarity > 0.3

    def test_generate_cached_skip(self, tmp_path: Path) -> None:
        source = tmp_path / "cache_test.parquet"
        df = pl.DataFrame({
            "id": ["c1"],
            "text": ["unique text"],
        })
        df.write_parquet(str(source))

        cfg = EmbeddingEngineConfig(
            source_paths={SourceType.observation: source},
            output_dir=tmp_path / "out3",
        )
        engine = EmbeddingEngine(cfg)

        r1 = engine.generate()
        assert r1["total_embedded"] == 1

        r2 = engine.generate()
        assert r2["total_skipped"] == 1  # cached
        assert r2["total_embedded"] == 0

        r3 = engine.generate(force=True)
        assert r3["total_embedded"] == 1  # re-computed


# ---------------------------------------------------------------------------
# Provider integration
# ---------------------------------------------------------------------------


def test_sentence_transformer_provider_creates() -> None:
    from phase2.embeddings.providers.sentence_transformers import SentenceTransformerProvider
    from phase2.embeddings.registry import get_provider_class

    cls = get_provider_class("sentence_transformers")
    assert cls is SentenceTransformerProvider

    cfg = EmbeddingEngineConfig(provider="sentence_transformers")
    provider = cls(cfg)
    assert provider.provider_name == "sentence_transformers"
    assert provider.dimension == 384

    vec = provider.embed_one("hello world")
    assert vec.shape == (384,)
    assert abs(np.linalg.norm(vec) - 1.0) < 0.01


def test_sentence_transformer_batch() -> None:
    from phase2.embeddings.registry import create_provider

    cfg = EmbeddingEngineConfig(provider="sentence_transformers")
    provider = create_provider(cfg)
    vecs = provider.embed(["a", "b", "c"])
    assert len(vecs) == 3
    for v in vecs:
        assert abs(np.linalg.norm(v) - 1.0) < 0.01