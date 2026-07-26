"""Phase 3, Module 2 — Knowledge Graph Reasoning Engine."""

from __future__ import annotations

from phase2.reasoning.schema import (
    Explanation,
    ExplanationFormat,
    ExplainabilityScore,
    EvidenceAggregation,
    InferenceOutput,
    InferenceResult,
    InferenceType,
    PropagationStrategy,
    ProvenanceVersion,
    ReasoningChain,
    ReasoningMetadata,
    ReasoningStep,
    RootCause,
    RootCauseRanking,
)
from phase2.reasoning.config import ReasoningConfig, load_reasoning_config
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.rule_engine import RuleEngine
from phase2.reasoning.chains import ChainTracker
from phase2.reasoning.evidence import EvidenceAggregator
from phase2.reasoning.inference import InferenceEngine
from phase2.reasoning.root_cause import RootCauseDiscoverer
from phase2.reasoning.explanation import ExplanationGenerator
from phase2.reasoning.cache import ReasoningCache
from phase2.reasoning.store import ReasoningStore
from phase2.reasoning.engine import ReasoningEngine

__all__ = [
    "InferenceType",
    "PropagationStrategy",
    "RootCauseRanking",
    "ExplanationFormat",
    "ReasoningStep",
    "InferenceResult",
    "ReasoningChain",
    "RootCause",
    "EvidenceAggregation",
    "Explanation",
    "ReasoningMetadata",
    "InferenceOutput",
    "ExplainabilityScore",
    "ProvenanceVersion",
    "ReasoningConfig",
    "load_reasoning_config",
    "ConfidencePropagator",
    "RuleEngine",
    "ChainTracker",
    "EvidenceAggregator",
    "InferenceEngine",
    "RootCauseDiscoverer",
    "ExplanationGenerator",
    "ReasoningCache",
    "ReasoningStore",
    "ReasoningEngine",
]

__version__ = "1.0.0"
