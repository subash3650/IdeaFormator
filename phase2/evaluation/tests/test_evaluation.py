from __future__ import annotations

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest

from phase2.evaluation.clusters import ClusterEvaluator
from phase2.evaluation.document import DocumentEvaluator
from phase2.evaluation.embeddings import EmbeddingEvaluator
from phase2.evaluation.evaluator import EvaluationOrchestrator
from phase2.evaluation.evidence import EvidenceEvaluator
from phase2.evaluation.exporter import export_all
from phase2.evaluation.observation import ObservationEvaluator
from phase2.evaluation.relationships import RelationshipEvaluator
from phase2.evaluation.reports import evaluation_to_dict, generate_summary
from phase2.evaluation.schema import GlobalEvaluation
from phase2.evaluation.signals import SignalEvaluator


# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def sample_documents() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [f"doc_{i}" for i in range(100)],
        "platform": ["google_play"] * 50 + ["amazon"] * 30 + ["yelp"] * 20,
        "source_dataset": ["dataset1"] * 60 + ["dataset2"] * 40,
        "title": [f"Title {i}" for i in range(100)],
        "text": [f"Document text content number {i} " * 5 for i in range(100)],
        "clean_text": [f"Clean text {i}" for i in range(100)],
        "rating": [float(i % 5) for i in range(100)],
        "author": [f"author_{i % 10}" for i in range(100)],
        "country": ["US"] * 40 + ["GB"] * 30 + ["IN"] * 20 + [None] * 10,
        "location": [""] * 100,
        "language": ["en"] * 80 + ["es"] * 10 + ["fr"] * 10,
        "created_at": [f"2024-{(i % 9) + 1:02d}-{(i % 28) + 1:02d}T00:00:00" for i in range(100)],
        "metadata": ['{"key": "val"}'] * 100,
        "raw_record": ['{}'] * 100,
        "document_length": [len(f"Document text content number {i} " * 5) for i in range(100)],
    })


@pytest.fixture
def sample_documents_with_duplicates() -> pl.DataFrame:
    ids = ["doc_0"] * 5 + [f"doc_{i}" for i in range(1, 95)]
    return pl.DataFrame({
        "id": ids,
        "platform": ["google_play"] * len(ids),
        "text": [f"Content {i}" for i in range(len(ids))],
        "language": ["en"] * len(ids),
        "created_at": ["2024-01-01"] * len(ids),
    })


@pytest.fixture
def sample_observations() -> pl.DataFrame:
    return pl.DataFrame({
        "observation_id": [f"obs_{i}" for i in range(50)],
        "type": ["entity"] * 15 + ["keyword"] * 12 + ["phrase"] * 10 + ["pattern"] * 8 + ["bigram"] * 5,
        "value": [f"value_{i % 20}" for i in range(50)],
        "document_id": [f"doc_{i % 10}" for i in range(50)],
        "platform": ["google_play"] * 50,
        "rating": [float(i % 5) for i in range(50)],
        "country": ["US"] * 50,
        "text_snippet": ["snippet"] * 50,
        "extractor": ["entity_extractor"] * 15 + ["keyword_extractor"] * 12 + ["phrase_extractor"] * 10 + ["pattern_matcher"] * 8 + ["ngram_extractor"] * 5,
        "method": ["dictionary_match"] * 50,
        "confidence": [0.85] * 50,
        "entity": [f"entity_{i % 8}" for i in range(50)],
        "entity_type": ["feature"] * 25 + ["bug"] * 25,
        "category": ["performance"] * 20 + ["usability"] * 20 + ["reliability"] * 10,
        "pattern_label": [""] * 50,
        "canonical_value": [f"canon_{i % 12}" for i in range(50)],
        "canonical_source": ["taxonomy"] * 50,
        "pipeline_version": ["1.5.0"] * 50,
        "generated_at": ["2024-01-01"] * 50,
    })


@pytest.fixture
def sample_evidence() -> pl.DataFrame:
    return pl.DataFrame({
        "evidence_id": [f"ev_{i}" for i in range(10)],
        "signal_key": [f"signal_{i}" for i in range(10)],
        "category": ["performance"] * 4 + ["usability"] * 3 + ["reliability"] * 3,
        "entity": [f"entity_{i % 5}" for i in range(10)],
        "entity_type": ["feature"] * 6 + ["bug"] * 4,
        "signal_text": [f"Signal text {i}" for i in range(10)],
        "observation_count": [5, 8, 3, 12, 6, 4, 10, 7, 9, 11],
        "document_count": [3, 5, 2, 8, 4, 3, 6, 5, 7, 6],
        "avg_rating": [2.5, 1.8, 3.0, 2.0, 2.2, 3.5, 1.5, 2.8, 2.1, 1.9],
        "platform_distribution": ['{"google_play": 5}'] * 10,
        "country_distribution": ['{"US": 5}'] * 10,
        "observation_ids": ["[obs1,obs2]"] * 10,
        "top_snippets": ["[snippet]"] * 10,
        "confidence": [0.85, 0.90, 0.70, 0.95, 0.80, 0.65, 0.88, 0.92, 0.78, 0.85],
        "aggregation_strategy": ["rule"] * 10,
        "pipeline_version": ["1.5.0"] * 10,
        "generated_at": ["2024-01-01"] * 10,
    })


@pytest.fixture
def sample_problem_signals() -> pl.DataFrame:
    return pl.DataFrame({
        "signal_key": [f"sig_{i}" for i in range(5)],
        "category": ["performance"] * 2 + ["usability"] * 2 + ["reliability"] * 1,
        "entity": [f"entity_{i}" for i in range(5)],
        "entity_type": ["feature"] * 3 + ["bug"] * 2,
        "country": [""] * 5,
        "signal_text": [f"Problem signal {i}" for i in range(5)],
        "document_count": [10, 8, 15, 6, 12],
        "avg_rating": [2.0, 1.5, 2.5, 3.0, 1.8],
        "evidence_ids": ["[ev1,ev2]"] * 5,
        "observation_count": [20, 15, 30, 10, 25],
        "confidence": [0.85, 0.75, 0.90, 0.70, 0.80],
        "pipeline_version": ["1.5.0"] * 5,
        "generated_at": ["2024-01-01"] * 5,
    })


@pytest.fixture
def sample_embeddings() -> pl.DataFrame:
    import numpy as np
    dim = 384
    np.random.seed(42)
    return pl.DataFrame({
        "embedding_id": [f"emb_{i}" for i in range(30)],
        "source_id": [f"src_{i}" for i in range(30)],
        "source_type": ["observation"] * 15 + ["evidence"] * 10 + ["problem_signal"] * 5,
        "provider": ["sentence_transformers"] * 30,
        "model": ["all-MiniLM-L6-v2"] * 30,
        "model_version": ["1.0"] * 30,
        "dimension": [384] * 30,
        "embedding": [np.random.randn(dim).tolist() for _ in range(30)],
        "text_snippet": [f"snippet {i}" for i in range(30)],
        "created_at": ["2024-01-01"] * 30,
    })


@pytest.fixture
def sample_relationships() -> pl.DataFrame:
    return pl.DataFrame({
        "relationship_id": [f"rel_{i}" for i in range(20)],
        "source_type": ["observation"] * 20,
        "source_id": [f"src_{i % 8}" for i in range(20)],
        "target_type": ["observation"] * 20,
        "target_id": [f"tgt_{(i + 3) % 8}" for i in range(20)],
        "relationship_type": ["similar"] * 15 + ["duplicate"] * 3 + ["causes"] * 2,
        "similarity_score": [0.95, 0.85, 0.75, 0.65, 0.55, 0.90, 0.80, 0.70, 0.60, 0.50,
                             0.88, 0.78, 0.68, 0.58, 0.48, 0.92, 0.82, 0.72, 0.62, 0.52],
        "confidence": [0.90, 0.85, 0.80, 0.75, 0.70, 0.88, 0.82, 0.78, 0.72, 0.68,
                       0.86, 0.84, 0.76, 0.74, 0.66, 0.91, 0.83, 0.79, 0.73, 0.69],
        "metric": ["cosine"] * 20,
        "provider": ["sentence_transformers"] * 20,
        "model_fingerprint": ["all-MiniLM-L6-v2@384"] * 20,
        "shared_entities": [[]] * 20,
        "shared_categories": [[]] * 20,
        "support_count": [1] * 20,
        "metadata": ['{}'] * 20,
        "version": ["1.0.0"] * 20,
        "created_at": ["2024-01-01"] * 20,
    })


@pytest.fixture
def sample_clusters() -> pl.DataFrame:
    return pl.DataFrame({
        "cluster_id": [f"cluster_{i}" for i in range(6)],
        "representative_id": [f"rep_{i}" for i in range(6)],
        "member_ids": [[f"mem_{i}_{j}" for j in range(3)] for i in range(3)]
                     + [[f"mem_{i}"] for i in range(3, 6)],
        "member_count": [5, 3, 2, 1, 1, 1],
        "relationship_count": [10, 6, 3, 0, 0, 0],
        "average_similarity": [0.85, 0.75, 0.65, 0.0, 0.0, 0.0],
        "density": [0.8, 0.6, 0.5, 0.0, 0.0, 0.0],
        "quality_score": [0.9, 0.7, 0.5, 0.2, 0.1, 0.0],
        "cluster_type": ["normal"] * 3 + ["low_quality"] * 3,
        "provider": ["connected_components"] * 6,
        "provider_version": ["1.0.0"] * 6,
        "algorithm": ["connected_components"] * 6,
        "metadata": ['{}'] * 6,
        "version": ["1.0.0"] * 6,
        "created_at": ["2024-01-01"] * 6,
    })


# ── Document Evaluation Tests ────────────────────────────────────

class TestDocumentEvaluator:
    def test_evaluate_normal(self, sample_documents):
        ev = DocumentEvaluator().evaluate(sample_documents)
        assert ev.total_documents == 100
        assert ev.documents_per_source["google_play"] == 50
        assert ev.duplicate_rate == 0.0
        assert ev.avg_document_length > 0
        assert "en" in ev.language_distribution
        assert len(ev.missing_fields) > 0
        assert ev.empty_content_rate == 0.0
        assert ev.metadata_completeness > 0.8

    def test_evaluate_empty(self):
        empty = pl.DataFrame({"id": pl.Series([], dtype=pl.Utf8), "text": pl.Series([], dtype=pl.Utf8)})
        ev = DocumentEvaluator().evaluate(empty)
        assert ev.total_documents == 0
        assert ev.health.score == 0.0

    def test_evaluate_no_text_column(self):
        df = pl.DataFrame({"id": ["a", "b"], "platform": ["x", "y"]})
        ev = DocumentEvaluator().evaluate(df)
        assert ev.total_documents == 2
        assert ev.avg_document_length == 0.0

    def test_duplicate_detection(self, sample_documents_with_duplicates):
        ev = DocumentEvaluator().evaluate(sample_documents_with_duplicates)
        assert ev.duplicate_count == 4
        assert ev.duplicate_rate == pytest.approx(4 / 99, 0.01)

    def test_empty_content_detection(self):
        df = pl.DataFrame({
            "id": ["a", "b", "c"],
            "text": ["hello", "", None],
            "platform": ["x", "y", "z"],
        })
        ev = DocumentEvaluator().evaluate(df)
        assert ev.empty_content_count == 2
        assert ev.empty_content_rate == pytest.approx(2 / 3, 0.01)

    def test_date_range(self, sample_documents):
        ev = DocumentEvaluator().evaluate(sample_documents)
        assert ev.date_range.get("earliest") is not None, f"empty date_range: {ev.date_range}"
        assert ev.collection_freshness_days > 0, f"freshness: {ev.collection_freshness_days}"

    def test_health_score_deductions(self):
        df = pl.DataFrame({
            "id": ["a", "a"],
            "text": ["", None],
            "platform": ["x", "y"],
            "language": [None, None],
        })
        ev = DocumentEvaluator().evaluate(df)
        assert ev.health.score < 50

    def test_not_modifying_input(self, sample_documents):
        original = sample_documents.clone()
        DocumentEvaluator().evaluate(sample_documents)
        assert sample_documents.equals(original)

    def test_handles_missing_platform_column(self):
        df = pl.DataFrame({"id": ["a", "b"], "text": ["hello", "world"]})
        ev = DocumentEvaluator().evaluate(df)
        assert ev.documents_per_source == {}


# ── Observation Evaluation Tests ─────────────────────────────────

class TestObservationEvaluator:
    def test_evaluate_normal(self, sample_observations):
        ev = ObservationEvaluator().evaluate(sample_observations, doc_count=10)
        assert ev.total_observations == 50
        assert ev.total_documents == 10
        assert ev.entity_precision > 0
        assert ev.canonicalization_success_rate == 1.0
        assert ev.knowledge_enrichment_coverage == 1.0
        assert len(ev.extractor_contribution) >= 4
        assert ev.observations_per_document.mean > 0
        assert "entity" in ev.type_distribution

    def test_evaluate_empty(self):
        empty = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in ["observation_id", "type", "value"]})
        ev = ObservationEvaluator().evaluate(empty)
        assert ev.total_observations == 0
        assert ev.health.score == 0.0

    def test_evaluate_no_observations_file(self, tmp_path):
        ev = ObservationEvaluator(knowledge_dir=str(tmp_path)).evaluate()
        assert ev.total_observations == 0

    def test_entity_precision_no_entity_column(self):
        df = pl.DataFrame({
            "observation_id": ["o1", "o2"],
            "type": ["keyword", "keyword"],
            "value": ["v1", "v2"],
            "document_id": ["d1", "d2"],
        })
        ev = ObservationEvaluator().evaluate(df, doc_count=2)
        assert ev.entity_precision == 0.0

    def test_extractor_contribution(self, sample_observations):
        ev = ObservationEvaluator().evaluate(sample_observations)
        total = sum(ev.extractor_contribution.values())
        assert total == 50
        assert "entity_extractor" in ev.extractor_contribution

    def test_not_modifying_input(self, sample_observations):
        original = sample_observations.clone()
        ObservationEvaluator().evaluate(sample_observations)
        assert sample_observations.equals(original)


# ── Evidence Evaluation Tests ────────────────────────────────────

class TestEvidenceEvaluator:
    def test_evaluate_normal(self, sample_evidence):
        ev = EvidenceEvaluator().evaluate(sample_evidence, obs_count=50)
        assert ev.total_evidence == 10
        assert ev.compression_ratio == 5.0
        assert ev.support_distribution.mean > 0
        assert ev.evidence_confidence.mean > 0.7
        assert ev.avg_observations_per_evidence > 0
        assert len(ev.category_distribution) == 3
        assert "feature" in ev.entity_type_distribution

    def test_evaluate_empty(self):
        empty = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in ["evidence_id", "signal_key"]})
        ev = EvidenceEvaluator().evaluate(empty)
        assert ev.total_evidence == 0
        assert ev.health.score == 0.0

    def test_compression_ratio_with_zero_obs(self, sample_evidence):
        ev = EvidenceEvaluator().evaluate(sample_evidence, obs_count=0)
        assert ev.compression_ratio > 0

    def test_not_modifying_input(self, sample_evidence):
        original = sample_evidence.clone()
        EvidenceEvaluator().evaluate(sample_evidence)
        assert sample_evidence.equals(original)


# ── Signal Evaluation Tests ───────────────────────────────────────

class TestSignalEvaluator:
    def test_evaluate_normal(self, sample_problem_signals):
        ev = SignalEvaluator().evaluate(sample_problem_signals)
        assert ev.accepted_signals == 5
        assert ev.support_distribution.mean > 0
        assert ev.confidence_distribution.mean > 0.7
        assert len(ev.category_coverage) == 3

    def test_evaluate_empty(self):
        empty = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in ["signal_key", "category"]})
        ev = SignalEvaluator().evaluate(empty)
        assert ev.accepted_signals == 0
        assert ev.zero_signal_explanation
        assert ev.health.score == 0.0

    def test_not_modifying_input(self, sample_problem_signals):
        original = sample_problem_signals.clone()
        SignalEvaluator().evaluate(sample_problem_signals)
        assert sample_problem_signals.equals(original)


# ── Embedding Evaluation Tests ────────────────────────────────────

class TestEmbeddingEvaluator:
    def test_evaluate_normal(self, sample_embeddings, tmp_path):
        phase2_dir = tmp_path / "assets" / "phase2"
        phase2_dir.mkdir(parents=True, exist_ok=True)
        fname = phase2_dir / "embeddings_observation.parquet"
        sample_embeddings.write_parquet(str(fname))

        ev = EmbeddingEvaluator(knowledge_dir=str(tmp_path)).evaluate()
        assert ev.total_vectors == 30
        assert ev.dimension == 384
        assert ev.provider == "sentence_transformers"
        assert ev.model == "all-MiniLM-L6-v2"
        assert ev.vector_norm_distribution.mean > 0
        assert ev.zero_vector_count == 0

    def test_evaluate_no_files(self, tmp_path):
        ev = EmbeddingEvaluator(knowledge_dir=str(tmp_path)).evaluate()
        assert ev.total_vectors == 0
        assert ev.health.score == 0.0

    def test_duplicate_vector_detection(self, tmp_path):
        import numpy as np
        vec = np.random.randn(384).tolist()
        phase2_dir = tmp_path / "assets" / "phase2"
        phase2_dir.mkdir(parents=True)
        df = pl.DataFrame({
            "embedding_id": ["e1", "e2", "e3"],
            "source_id": ["s1", "s2", "s3"],
            "source_type": ["observation"] * 3,
            "provider": ["sp"] * 3,
            "model": ["m"] * 3,
            "dimension": [384] * 3,
            "embedding": [vec, vec, [0.0] * 384],
        })
        df.write_parquet(str(phase2_dir / "embeddings_observation.parquet"))
        ev = EmbeddingEvaluator(knowledge_dir=str(tmp_path)).evaluate()
        assert ev.duplicate_vector_count == 1
        assert ev.zero_vector_count == 1


# ── Relationship Evaluation Tests ────────────────────────────────

class TestRelationshipEvaluator:
    def test_evaluate_normal(self, sample_relationships):
        ev = RelationshipEvaluator().evaluate(sample_relationships)
        assert ev.total_relationships == 20
        assert ev.average_similarity > 0.5
        assert ev.similarity_distribution.mean > 0
        assert ev.confidence_distribution.mean > 0.7
        assert len(ev.relationship_type_distribution) == 3
        assert ev.degree_distribution.mean > 0

    def test_evaluate_empty(self):
        empty = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in ["relationship_id", "source_id"]})
        ev = RelationshipEvaluator().evaluate(empty)
        assert ev.total_relationships == 0
        assert ev.health.score == 0.0

    def test_connected_component_no_duplicate_edges(self):
        df = pl.DataFrame({
            "relationship_id": ["r1", "r2"],
            "source_type": ["obs", "obs"],
            "source_id": ["a", "b"],
            "target_type": ["obs", "obs"],
            "target_id": ["b", "a"],
            "similarity_score": [0.9, 0.9],
            "confidence": [0.9, 0.9],
        })
        ev = RelationshipEvaluator().evaluate(df)
        assert ev.largest_connected_component_size == 2

    def test_not_modifying_input(self, sample_relationships):
        original = sample_relationships.clone()
        RelationshipEvaluator().evaluate(sample_relationships)
        assert sample_relationships.equals(original)


# ── Cluster Evaluation Tests ─────────────────────────────────────

class TestClusterEvaluator:
    def test_evaluate_normal(self, sample_clusters):
        ev = ClusterEvaluator().evaluate(sample_clusters)
        assert ev.total_clusters == 6
        assert ev.total_members == 13
        assert ev.singleton_count == 3
        assert ev.singleton_rate == 0.5
        assert ev.low_quality_count == 3
        assert ev.low_quality_rate == 0.5
        assert len(ev.largest_clusters) == 5
        assert ev.cluster_type_distribution.get("normal") == 3
        assert ev.cluster_type_distribution.get("low_quality") == 3
        assert ev.quality_distribution.mean > 0
        assert ev.density_distribution.mean > 0

    def test_evaluate_empty(self):
        empty = pl.DataFrame({c: pl.Series([], dtype=pl.Utf8) for c in ["cluster_id", "member_count"]})
        ev = ClusterEvaluator().evaluate(empty)
        assert ev.total_clusters == 0
        assert ev.health.score == 0.0

    def test_largest_clusters_sorted(self, sample_clusters):
        ev = ClusterEvaluator().evaluate(sample_clusters)
        counts = [c["member_count"] for c in ev.largest_clusters]
        assert counts == sorted(counts, reverse=True)

    def test_not_modifying_input(self, sample_clusters):
        original = sample_clusters.clone()
        ClusterEvaluator().evaluate(sample_clusters)
        assert sample_clusters.equals(original)


# ── Health Score Tests ───────────────────────────────────────────

class TestHealthScores:
    def test_perfect_data_scores_high(self):
        df = pl.DataFrame({
            "id": [f"doc_{i}" for i in range(100)],
            "platform": ["google_play"] * 100,
            "text": ["Hello world"] * 100,
            "language": ["en"] * 100,
            "created_at": ["2024-01-01"] * 100,
        })
        doc_ev = DocumentEvaluator().evaluate(df)
        assert doc_ev.health.score >= 90

    def test_health_score_range(self):
        import math
        for score in [0, 25, 50, 75, 100]:
            assert 0 <= score <= 100


# ── Orchestrator Tests ───────────────────────────────────────────

class TestOrchestrator:
    def test_evaluate_with_real_assets(self):
        orchestrator = EvaluationOrchestrator()
        result = orchestrator.evaluate()
        assert isinstance(result, GlobalEvaluation)
        assert result.documents.total_documents >= 0
        assert result.observations.total_observations >= 0
        assert result.evidence.total_evidence >= 0
        assert result.signals.accepted_signals >= 0
        assert result.embeddings.total_vectors >= 0
        assert result.relationships.total_relationships >= 0
        assert result.clusters.total_clusters >= 0
        assert 0 <= result.overall_health_score <= 100
        assert result.generated_at

    def test_evaluate_with_missing_assets(self, tmp_path, monkeypatch):
        import pathlib
        _orig_exists = pathlib.Path.exists
        def _patched_exists(self):
            s = str(self)
            if "processed.parquet" in s or "knowledge" in s.lower():
                return False
            return _orig_exists(self)
        monkeypatch.setattr(pathlib.Path, "exists", _patched_exists)
        orchestrator = EvaluationOrchestrator(knowledge_dir=str(tmp_path / "empty_knowledge"))
        result = orchestrator.evaluate()
        assert result.documents.total_documents == 0, f"Got {result.documents.total_documents} docs"
        assert result.observations.total_observations == 0
        assert result.overall_health_score == 0.0

    def test_evaluate_with_known_structure(self, tmp_path):
        kb = tmp_path / "knowledge"
        processed = kb / "processed"
        assets = kb / "assets"
        processed.mkdir(parents=True)
        assets.mkdir(parents=True)

        pl.DataFrame({"id": ["a", "b"], "text": ["hello", "world"], "platform": ["x", "y"]}).write_parquet(
            str(processed / "processed.parquet")
        )
        pl.DataFrame({
            "observation_id": ["o1", "o2"], "type": ["entity", "keyword"],
            "value": ["v1", "v2"], "document_id": ["a", "b"],
            "platform": ["x", "y"], "rating": [1.0, 2.0],
            "country": ["US", "US"], "text_snippet": ["s", "s"],
            "extractor": ["e1", "e2"], "method": ["m", "m"],
            "confidence": [0.5, 0.5], "entity": ["ent1", "ent2"],
            "entity_type": ["feature", "bug"], "category": ["cat1", "cat2"],
            "pattern_label": ["", ""], "canonical_value": ["c1", "c2"],
            "canonical_source": ["tax", "tax"],
            "pipeline_version": ["1.5.0", "1.5.0"],
            "generated_at": ["2024-01-01", "2024-01-01"],
        }).write_parquet(str(assets / "observations.parquet"))

        orchestrator = EvaluationOrchestrator(knowledge_dir=str(kb))
        result = orchestrator.evaluate()
        assert result.documents.total_documents == 2
        assert result.observations.total_observations == 2
        assert result.overall_health_score > 0


# ── Reporter Tests ───────────────────────────────────────────────

class TestReporter:
    def test_evaluation_to_dict(self):
        ev = GlobalEvaluation()
        d = evaluation_to_dict(ev)
        assert isinstance(d, dict)
        assert "overall_health_score" in d

    def test_generate_summary(self):
        ev = GlobalEvaluation()
        s = generate_summary(ev)
        assert "overall_health_score" in s
        assert "stages" in s
        assert "recommendations" in s

    def test_summary_keys(self):
        ev = GlobalEvaluation()
        ev.overall_health_score = 85.0
        ev.documents.health.score = 90.0
        s = generate_summary(ev)
        assert s["overall_health_score"] == 85.0
        assert s["stages"]["documents"]["health_score"] == 90.0


# ── Exporter Tests ───────────────────────────────────────────────

class TestExporter:
    def test_export_all_creates_files(self):
        ev = EvaluationOrchestrator().evaluate()
        with TemporaryDirectory() as tmp:
            paths = export_all(ev, output_dir=tmp)
            assert "report" in paths
            assert "summary" in paths
            assert "dashboard_json" in paths
            assert "dashboard_txt" in paths
            for p in paths.values():
                assert Path(p).exists()
                assert Path(p).stat().st_size > 0

    def test_export_report_is_valid_json(self):
        ev = EvaluationOrchestrator().evaluate()
        with TemporaryDirectory() as tmp:
            paths = export_all(ev, output_dir=tmp)
            with open(paths["report"]) as f:
                data = json.load(f)
            assert "overall_health_score" in data
            assert "documents" in data

    def test_dashboard_txt_contains_health_score(self):
        from phase2.evaluation.schema import GlobalEvaluation
        ev = GlobalEvaluation()
        ev.overall_health_score = 75.0
        ev.documents.health.score = 80.0
        with TemporaryDirectory() as tmp:
            paths = export_all(ev, output_dir=tmp)
            text = Path(paths["dashboard_txt"]).read_text()
            assert "PAIN INTELLIGENCE" in text
            assert "75" in text
            assert "Score:" in text


# ── Metric Utility Tests ─────────────────────────────────────────

class TestMetrics:
    def test_safe_divide(self):
        from phase2.evaluation.metrics import safe_divide
        assert safe_divide(10, 2) == 5.0
        assert safe_divide(10, 0) == 0.0

    def test_compute_distribution(self):
        from phase2.evaluation.metrics import compute_distribution
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        d = compute_distribution(vals, bins=3)
        assert d.min == 1.0
        assert d.max == 5.0
        assert d.mean == 3.0
        assert d.median == 3.0
        assert len(d.histogram) > 0

    def test_compute_distribution_empty(self):
        from phase2.evaluation.metrics import compute_distribution
        d = compute_distribution([])
        assert d.min == 0.0

    def test_uniqueness_ratio(self):
        from phase2.evaluation.metrics import uniqueness_ratio
        s = pl.Series("x", ["a", "b", "c", "a"])
        assert uniqueness_ratio(s) == 0.75

    def test_entropy(self):
        from phase2.evaluation.metrics import entropy
        s = pl.Series("x", ["a", "a", "b", "b"])
        e = entropy(s)
        assert e == pytest.approx(1.0, 0.01)

    def test_value_counts(self):
        from phase2.evaluation.metrics import value_counts
        s = pl.Series("x", ["a", "a", "b"])
        assert value_counts(s) == {"a": 2, "b": 1}

    def test_column_exists(self):
        from phase2.evaluation.metrics import column_exists
        df = pl.DataFrame({"a": [1]})
        assert column_exists(df, "a")
        assert not column_exists(df, "b")

    def test_null_count(self):
        from phase2.evaluation.metrics import null_count
        s = pl.Series("x", [1, None, 3])
        assert null_count(s) == 1


# ── Edge Cases ───────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_dataframe_for_each_evaluator(self):
        empty = pl.DataFrame()
        for evaluator_cls, kwargs in [
            (DocumentEvaluator, {}),
            (ObservationEvaluator, {}),
            (EvidenceEvaluator, {}),
            (SignalEvaluator, {}),
            (ClusterEvaluator, {}),
            (RelationshipEvaluator, {}),
        ]:
            ev = evaluator_cls().evaluate(**kwargs) if not kwargs else evaluator_cls(**kwargs).evaluate(empty)
            assert hasattr(ev, "health")

    def test_single_document(self):
        df = pl.DataFrame({
            "id": ["doc_1"],
            "platform": ["google_play"],
            "text": ["Hello"],
            "language": ["en"],
        })
        ev = DocumentEvaluator().evaluate(df)
        assert ev.total_documents == 1
        assert ev.health.score >= 50

    def test_mixed_missing_columns(self):
        df = pl.DataFrame({
            "id": ["a", "b", "c"],
        })
        ev = DocumentEvaluator().evaluate(df)
        assert ev.total_documents == 3
        assert ev.avg_document_length == 0.0
        assert ev.empty_content_count == 0

    def test_large_file_does_not_timeout(self):
        import numpy as np
        n = 1000
        df = pl.DataFrame({
            "id": [f"doc_{i}" for i in range(n)],
            "platform": ["google_play"] * n,
            "text": ["Hello world " * 10] * n,
            "language": ["en"] * n,
            "created_at": ["2024-01-01"] * n,
            "rating": [float(i % 5) for i in range(n)],
        })
        import time
        t0 = time.time()
        ev = DocumentEvaluator().evaluate(df)
        elapsed = time.time() - t0
        assert elapsed < 5.0
        assert ev.total_documents == n


# ── Determinism Tests ────────────────────────────────────────────

class TestDeterminism:
    def test_same_input_produces_same_output(self, sample_documents):
        ev1 = DocumentEvaluator().evaluate(sample_documents.clone())
        ev2 = DocumentEvaluator().evaluate(sample_documents.clone())
        assert ev1.health.score == ev2.health.score
        assert ev1.duplicate_rate == ev2.duplicate_rate

    def test_observation_evaluator_determinism(self, sample_observations):
        ev1 = ObservationEvaluator().evaluate(sample_observations.clone(), doc_count=10)
        ev2 = ObservationEvaluator().evaluate(sample_observations.clone(), doc_count=10)
        assert ev1.health.score == ev2.health.score

    def test_embedding_evaluator_determinism(self, sample_embeddings, tmp_path):
        d1 = tmp_path / "r1"
        d2 = tmp_path / "r2"
        for d in [d1, d2]:
            (d / "assets" / "phase2").mkdir(parents=True)
            sample_embeddings.clone().write_parquet(str(d / "assets" / "phase2" / "embeddings_observation.parquet"))
        ev1 = EmbeddingEvaluator(knowledge_dir=str(d1)).evaluate()
        ev2 = EmbeddingEvaluator(knowledge_dir=str(d2)).evaluate()
        assert ev1.total_vectors == ev2.total_vectors


# ── No-Modification Tests ────────────────────────────────────────

class TestNoModification:
    def test_orchestrator_does_not_modify_assets(self):
        import os
        knowledge = Path("pain_intelligence/knowledge")
        assets_dir = knowledge / "assets"
        if not assets_dir.exists():
            pytest.skip("No assets directory")
        snapshots = {}
        for f in assets_dir.rglob("*.parquet"):
            snapshots[str(f)] = (f.stat().st_size, os.path.getmtime(f))
        EvaluationOrchestrator().evaluate()
        for f_path, (size, mtime) in snapshots.items():
            p = Path(f_path)
            assert p.stat().st_size == size
            assert os.path.getmtime(p) == mtime
