"""Inference engine — multi-pass reasoning over the knowledge graph."""

from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.reasoning.chains import ChainTracker
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.evidence import EvidenceAggregator
from phase2.reasoning.explanation import ExplanationGenerator
from phase2.reasoning.config import ReasoningConfig
from phase2.reasoning.root_cause import RootCauseDiscoverer
from phase2.reasoning.rule_engine import RuleEngine
from phase2.reasoning.schema import (
    InferenceOutput,
    InferenceResult,
    ReasoningMetadata,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class InferenceEngine:
    def __init__(
        self,
        config: ReasoningConfig,
        rule_engine: RuleEngine,
        propagator: ConfidencePropagator,
        chain_tracker: ChainTracker,
        evidence_aggregator: EvidenceAggregator,
        root_cause_discoverer: RootCauseDiscoverer,
        explanation_generator: ExplanationGenerator,
    ) -> None:
        self._config = config
        self._rule_engine = rule_engine
        self._propagator = propagator
        self._chain_tracker = chain_tracker
        self._evidence_aggregator = evidence_aggregator
        self._root_cause_discoverer = root_cause_discoverer
        self._explanation_generator = explanation_generator

    def infer(self, graph: GraphInterface, run_id: str) -> InferenceOutput:
        start = time.perf_counter()

        self._rule_engine.initialize_rules(graph)

        rule_inferences = self._rule_engine.apply_all(graph, self._propagator)
        rule_inferences = rule_inferences[: self._config.max_inferences_per_run]

        for inf in rule_inferences:
            chain_id = self._chain_tracker.start_chain(
                inference_id=inf.inference_id,
                input_node_ids=inf.provenance,
            )
            self._chain_tracker.add_step(
                chain_id=chain_id,
                rule_name=inf.inference_type.value,
                input_node_ids=inf.provenance,
                output_node_id=inf.derived_node_id,
                output_edge_id=inf.derived_edge_id,
                confidence_delta=inf.confidence,
            )
            output_node_ids = [inf.derived_node_id] if inf.derived_node_id else []
            output_edge_ids = [inf.derived_edge_id] if inf.derived_edge_id else []
            self._chain_tracker.finalize(
                chain_id=chain_id,
                total_confidence=inf.confidence,
                output_node_ids=output_node_ids,
                output_edge_ids=output_edge_ids,
            )
            inf = inf.model_copy(update={"chain_id": chain_id})

        all_inferences = rule_inferences
        chains = self._chain_tracker.all_chains()

        root_causes = self._root_cause_discoverer.discover(
            graph, propagator=self._propagator
        )

        evidence_aggregations = self._evidence_aggregator.aggregate_all(
            graph, propagator=self._propagator
        )

        explanations: list = []
        if self._config.generate_explanations:
            for inf in all_inferences[:50]:
                chain = self._chain_tracker.get_chain(inf.chain_id)
                if chain:
                    expl = self._explanation_generator.explain_inference(
                        inf, chain, graph,
                        format=self._config.explanation_format,
                        collapse_threshold=self._config.collapse_chains_longer_than,
                    )
                    explanations.append(expl)

        rule_firing: dict[str, int] = {}
        for inf in all_inferences:
            key = inf.inference_type.value
            rule_firing[key] = rule_firing.get(key, 0) + 1

        elapsed = time.perf_counter() - start
        meta = ReasoningMetadata(
            run_id=run_id,
            kg_run_id="",
            inference_count=len(all_inferences),
            chain_count=len(chains),
            root_cause_count=len(root_causes),
            explanation_count=len(explanations),
            rules_applied=list(rule_firing.keys()),
            rule_firing_counts=rule_firing,
            elapsed_seconds=round(elapsed, 4),
            cache_hit=False,
        )

        return InferenceOutput(
            inferences=all_inferences,
            chains=chains,
            root_causes=root_causes,
            evidence_aggregations=evidence_aggregations,
            explanations=explanations,
            metadata=meta,
        )
