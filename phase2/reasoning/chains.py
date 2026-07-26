"""Reasoning chain tracker — full provenance for every inference."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from phase2.reasoning.schema import (
    ProvenanceVersion,
    ReasoningChain,
    ReasoningStep,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ChainTracker:
    def __init__(self, run_id: str = "") -> None:
        self._chains: dict[str, ReasoningChain] = {}
        self._current_steps: dict[str, list[ReasoningStep]] = {}
        self._run_id = run_id

    def start_chain(
        self,
        inference_id: str,
        input_node_ids: list[str] | None = None,
        provenance_version: ProvenanceVersion | None = None,
    ) -> str:
        chain_id = hashlib.sha256(
            f"chain:{inference_id}:{_now_iso()}:{self._run_id}".encode()
        ).hexdigest()
        chain = ReasoningChain(
            chain_id=chain_id,
            inference_id=inference_id,
            input_node_ids=input_node_ids or [],
            provenance_version=provenance_version or ProvenanceVersion(run_id=self._run_id),
        )
        self._chains[chain_id] = chain
        self._current_steps[chain_id] = []
        return chain_id

    def add_step(
        self,
        chain_id: str,
        rule_name: str,
        rule_version: str = "1.0",
        input_node_ids: list[str] | None = None,
        input_edge_ids: list[str] | None = None,
        output_node_id: str | None = None,
        output_edge_id: str | None = None,
        confidence_delta: float = 0.0,
    ) -> ReasoningStep:
        steps = self._current_steps.get(chain_id, [])
        step_id = len(steps)
        step = ReasoningStep(
            step_id=step_id,
            rule_name=rule_name,
            rule_version=rule_version,
            input_node_ids=input_node_ids or [],
            input_edge_ids=input_edge_ids or [],
            output_node_id=output_node_id,
            output_edge_id=output_edge_id,
            confidence_delta=round(confidence_delta, 4),
        )
        steps.append(step)
        return step

    def finalize(
        self,
        chain_id: str,
        total_confidence: float,
        output_node_ids: list[str] | None = None,
        output_edge_ids: list[str] | None = None,
    ) -> ReasoningChain:
        if chain_id not in self._chains:
            raise ValueError(f"Unknown chain: {chain_id}")
        steps = self._current_steps.get(chain_id, [])
        chain = self._chains[chain_id]
        updated = chain.model_copy(update={
            "steps": steps,
            "output_node_ids": output_node_ids or [],
            "output_edge_ids": output_edge_ids or [],
            "total_confidence": round(total_confidence, 4),
        })
        self._chains[chain_id] = updated
        return updated

    def get_chain(self, chain_id: str) -> ReasoningChain | None:
        return self._chains.get(chain_id)

    def get_chains_for_inference(self, inference_id: str) -> list[ReasoningChain]:
        return [c for c in self._chains.values() if c.inference_id == inference_id]

    def all_chains(self) -> list[ReasoningChain]:
        return list(self._chains.values())

    def chain_count(self) -> int:
        return len(self._chains)
