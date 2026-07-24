"""Semantic Relationship Engine for IdeaFormator structured knowledge.

Architecture
============
::

    Embeddings (Parquet)
          │
          ▼
     VectorIndex          SimilarityProvider
     (nearest-neighbour)  (similarity metric)
          │                      │
          └──────────┬───────────┘
                     ▼
           RelationshipBuilder
           (deterministic IDs)
                     │
                     ▼
            ConfidencePolicy
            (weighted factors)
                     │
                     ▼
             FilterPipeline
         (6 composable stages)
                     │
                     ▼
           SemanticRelationshipStore
                  (Parquet)
                     │
                     ▼
          ┌──────────┼──────────┐
          ▼          ▼          ▼
      Manifest   Text/JSON    RelationshipSearcher
                  Reports     (query by source/target)

Component Responsibilities
==========================
*Schema* (``schema.py``)
    Frozen Pydantic models for ``SemanticRelationship``, ``RelationshipJob``,
    and ``RelationshipManifest``.  Defines the asset contract.

*Config* (``config.py``)
    ``SimilarityEngineConfig`` with defaults loaded from YAML.
    Controls metric, thresholds, batch size, allowed cross-type pairs, etc.

*Providers* (``providers/``)
    Pluggable similarity metrics.  Each provider implements
    ``SimilarityProvider(ABC)`` and is registered via the ``@register``
    decorator.  Built-in: ``cosine``, ``dot_product``, ``euclidean``.

*Indexes* (``indexes/``)
    Vector index abstractions for nearest-neighbour search.
    ``VectorIndex(ABC)`` defines the interface; ``LinearIndex`` is the
    exact brute-force implementation suitable for ≤100k vectors.

*Builder* (``builder.py``)
    Constructs frozen ``SemanticRelationship`` objects with deterministic
    SHA-256 relationship IDs based on ``sorted(source_id, target_id)``.

*Confidence* (``confidence.py``)
    Weighted geometric mean of available signals: similarity, frequency,
    quality, support count, metadata completeness.  Gracefully falls back
    when factors are unavailable.

*Filters* (``filters.py``)
    Six composable stages in order: ``SelfSimilarityFilter`` →
    ``DuplicateRelationshipFilter`` → ``ThresholdFilter`` →
    ``ConfidenceFilter`` → ``RelationshipPolicyFilter`` →
    ``TopKPerSourceFilter``.

*Store* (``store.py``)
    Reads and writes ``semantic_relationships.parquet``.  Provides
    ``save``, ``append``, ``load``, ``load_df``, ``count``, ``exists``.

*Search* (``search.py``)
    ``RelationshipSearcher`` queries stored relationships by source_id,
    target_id, or pair.  Never embeds arbitrary text.

*Metrics* (``metrics.py``)
    ``compute_stats()`` returns ``RelationshipStats``: counts, similarity
    and confidence aggregates, density.

*Statistics* (``statistics.py``)
    ``compute_relationship_statistics()`` returns ``RelationshipStatistics``:
    histograms, degree distribution, isolated/connected nodes, top concepts.

*Threshold* (``threshold.py``)
    ``ThresholdRecommender`` analyses pre-filter score distribution and
    suggests candidate thresholds with estimated relationship counts.

*Exporter* (``exporter.py``)
    ``write_manifest()``, ``generate_quality_report()`` (text),
    ``write_json_report()`` (JSON).  Includes threshold recommendations
    and filter pipeline statistics.

*Engine* (``engine.py``)
    ``SimilarityEngine`` orchestrates the full pipeline: load embeddings
    → VectorIndex → pairwise search → build → confidence → filter → store
    → manifest → reports.

*Pipeline* (``pipeline.py``)
    Thin wrapper around ``SimilarityEngine`` for batch orchestration.

Extension Points
================
New SimilarityProvider
----------------------
1. Create a file in ``providers/``.
2. Subclass ``SimilarityProvider``, implement ``compute_pairwise``,
   ``compute_scores``, ``name``, ``version``.
3. Decorate with ``@register("your_metric_name")``.

New VectorIndex
---------------
1. Create a file in ``indexes/``.
2. Subclass ``VectorIndex``, implement ``search``, ``search_batch``,
   ``size``, ``dimension``.
3. Optionally override ``save`` and ``load`` for persistence.
4. Export from ``indexes/__init__.py``.

Provider Registration
---------------------
Providers self-register via the ``@register`` decorator at import time.
The ``providers/__init__.py`` imports all provider modules to trigger
registration.  The ``SimilarityEngine`` top-level ``__init__.py``
imports ``providers`` for the same reason.

Configuration
=============
See ``configs/default.yaml`` for the full configuration reference.
Key values can be overridden via the YAML ``similarity:`` section.

Incremental Processing
======================
The engine checks for existing relationships before generation.
Use ``--force`` / ``force=True`` to regenerate.  The store supports
``append()`` for incremental additions, though current pipeline
regenerates from scratch to maintain consistency.
"""

from phase2.similarity import providers  # noqa: F401 - register providers