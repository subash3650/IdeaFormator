"""Explanation generation — human-readable reasoning explanations."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.reasoning.schema import (
    ExplainabilityScore,
    Explanation,
    ExplanationFormat,
    InferenceResult,
    ReasoningChain,
    ReasoningStep,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExplanationGenerator:
    def explain_inference(
        self,
        inference: InferenceResult,
        chain: ReasoningChain,
        graph: GraphInterface,
        format: ExplanationFormat = ExplanationFormat.TEMPLATE,
        collapse_threshold: int = 4,
    ) -> Explanation:
        inference_type = inference.inference_type.value
        steps_text: list[str] = []
        collapsed = 0

        nodes_map: dict[str, str] = {}
        for nid in chain.input_node_ids:
            node = graph.get_node(nid)
            nodes_map[nid] = node.label if node and node.label else nid
        for nid in chain.output_node_ids:
            node = graph.get_node(nid)
            nodes_map[nid] = node.label if node and node.label else nid
        for step in chain.steps:
            for nid in step.input_node_ids:
                if nid not in nodes_map:
                    node = graph.get_node(nid)
                    nodes_map[nid] = node.label if node and node.label else nid
            if step.output_node_id and step.output_node_id not in nodes_map:
                node = graph.get_node(step.output_node_id)
                nodes_map[step.output_node_id] = node.label if node and node.label else step.output_node_id

        raw_steps: list[str] = []
        for step in chain.steps:
            inputs = ", ".join(nodes_map.get(n, n) for n in step.input_node_ids[:3])
            extra = f" (+{len(step.input_node_ids) - 3} more)" if len(step.input_node_ids) > 3 else ""
            if step.output_node_id:
                output_label = nodes_map.get(step.output_node_id, step.output_node_id)
                raw_steps.append(
                    f"Rule '{step.rule_name}' applied to [{inputs}{extra}] → {output_label}"
                    f" (confidence: {step.confidence_delta:.2f})"
                )
            elif step.output_edge_id:
                raw_steps.append(
                    f"Rule '{step.rule_name}' applied to [{inputs}{extra}] → edge {step.output_edge_id}"
                    f" (confidence: {step.confidence_delta:.2f})"
                )
            else:
                raw_steps.append(
                    f"Rule '{step.rule_name}' applied to [{inputs}{extra}]"
                    f" (confidence: {step.confidence_delta:.2f})"
                )

        if len(raw_steps) > collapse_threshold:
            steps_text = raw_steps[:2]
            collapsed = len(raw_steps) - 4
            steps_text.append(f"... {collapsed} intermediate reasoning steps ...")
            steps_text.extend(raw_steps[-2:])
        else:
            steps_text = raw_steps

        title = self._generate_title(inference, nodes_map)
        summary = self._generate_summary(inference, chain, nodes_map)
        evidence_summary = self._generate_evidence_summary(inference, chain)
        confidence_text = self._generate_confidence_explanation(inference, chain)

        reasoning_depth = len(chain.steps)
        evidence_count = max(1, sum(len(s.input_node_ids) for s in chain.steps))
        exp_score = ExplainabilityScore.compute(
            confidence=inference.confidence,
            evidence_count=evidence_count,
            reasoning_depth=reasoning_depth,
        )

        raw_text = self._render_text(title, summary, steps_text, evidence_summary, confidence_text)

        explanation_id = hashlib.sha256(
            f"explain:{inference.inference_id}:{format.value}:1.0".encode()
        ).hexdigest()

        return Explanation(
            explanation_id=explanation_id,
            inference_id=inference.inference_id,
            format=format,
            title=title,
            summary=summary,
            steps=steps_text,
            collapsed_step_count=collapsed,
            evidence_summary=evidence_summary,
            confidence_explanation=confidence_text,
            explainability_score=exp_score,
            raw_text=raw_text,
        )

    def _generate_title(self, inference: InferenceResult, nodes_map: dict[str, str]) -> str:
        if inference.inference_type.value == "transitive":
            if inference.provenance and len(inference.provenance) >= 3:
                src = nodes_map.get(inference.provenance[0], inference.provenance[0])
                tgt = nodes_map.get(inference.provenance[-1], inference.provenance[-1])
                return f"Transitive Inference: {src} → {tgt}"
            return "Transitive Edge Inference"
        if inference.inference_type.value == "causal_chain":
            if inference.provenance and len(inference.provenance) >= 2:
                src = nodes_map.get(inference.provenance[0], inference.provenance[0])
                tgt = nodes_map.get(inference.provenance[-1], inference.provenance[-1])
                return f"Causal Chain: {src} → ... → {tgt}"
            return "Causal Chain Inference"
        if inference.inference_type.value == "evidence_aggregation":
            nid = inference.derived_node_id or ""
            label = nodes_map.get(nid, nid)
            return f"Evidence Aggregation: {label}"
        return "Reasoning Inference"

    def _generate_summary(self, inference: InferenceResult, chain: ReasoningChain, nodes_map: dict[str, str]) -> str:
        parts: list[str] = []
        if inference.inference_type.value == "transitive":
            src = nodes_map.get(inference.provenance[0], inference.provenance[0]) if inference.provenance else "?"
            tgt = nodes_map.get(inference.provenance[-1], inference.provenance[-1]) if inference.provenance else "?"
            parts.append(
                f"Derived transitive relationship: {src} influences {tgt} "
                f"through a chain of {len(inference.provenance) - 1} hops."
            )
        elif inference.inference_type.value == "causal_chain":
            parts.append(
                f"Discovered causal chain with {len(inference.provenance)} nodes "
                f"and {chain.total_confidence:.2f} propagated confidence."
            )
        elif inference.inference_type.value == "evidence_aggregation":
            parts.append(
                f"Aggregated {len(chain.input_node_ids)} evidence sources "
                f"with combined confidence {chain.total_confidence:.2f}."
            )
        parts.append(f"Total reasoning steps: {len(chain.steps)}.")
        return " ".join(parts)

    def _generate_evidence_summary(self, inference: InferenceResult, chain: ReasoningChain) -> str:
        all_inputs: set[str] = set()
        for step in chain.steps:
            all_inputs.update(step.input_node_ids)
        return f"Uses {len(all_inputs)} unique input nodes across {len(chain.steps)} reasoning steps."

    def _generate_confidence_explanation(self, inference: InferenceResult, chain: ReasoningChain) -> str:
        return (
            f"Final confidence {inference.confidence:.2f} "
            f"propagated through {len(chain.steps)} steps."
        )

    def _render_text(
        self,
        title: str,
        summary: str,
        steps: list[str],
        evidence_summary: str,
        confidence_text: str,
    ) -> str:
        lines = [
            f"# {title}",
            "",
            summary,
            "",
            "## Reasoning Steps",
        ]
        for i, step in enumerate(steps, 1):
            lines.append(f"  {i}. {step}")
        lines.extend([
            "",
            "## Evidence",
            f"  {evidence_summary}",
            "",
            "## Confidence",
            f"  {confidence_text}",
        ])
        return "\n".join(lines)

    def explain_root_cause(
        self,
        root_cause,
        graph: GraphInterface,
        format: ExplanationFormat = ExplanationFormat.TEMPLATE,
    ) -> Explanation:
        title = f"Root Cause: {root_cause.cause_label}"
        summary = (
            f"{root_cause.cause_label} is a root cause of {root_cause.effect_label} "
            f"({root_cause.path_length} hops, confidence {root_cause.propagated_confidence:.2f}, "
            f"{root_cause.transitive_impact_count} downstream effects)."
        )
        path_labels: list[str] = []
        for nid in root_cause.path:
            node = graph.get_node(nid)
            path_labels.append(node.label if node and node.label else nid)
        steps_text = [
            f"Causal path: {' → '.join(path_labels)}",
            f"Propagated confidence: {root_cause.propagated_confidence:.2f}",
            f"Transitive impact: {root_cause.transitive_impact_count} downstream effects",
            f"Evidence count: {root_cause.evidence_count}",
        ]

        reasoning_depth = root_cause.path_length
        exp_score = ExplainabilityScore.compute(
            confidence=root_cause.propagated_confidence,
            evidence_count=root_cause.evidence_count,
            reasoning_depth=reasoning_depth,
        )
        raw_text = self._render_text(title, summary, steps_text, "", "")

        explanation_id = hashlib.sha256(
            f"explain:root_cause:{root_cause.cause_node_id}:{root_cause.effect_node_id}:{format.value}".encode()
        ).hexdigest()

        return Explanation(
            explanation_id=explanation_id,
            inference_id="",
            format=format,
            title=title,
            summary=summary,
            steps=steps_text,
            evidence_summary=f"Based on {root_cause.evidence_count} evidence items.",
            confidence_explanation=f"Confidence {root_cause.propagated_confidence:.2f} via {root_cause.path_length}-hop path.",
            explainability_score=exp_score,
            raw_text=raw_text,
        )

    def explain_evidence(
        self,
        aggregation,
        graph: GraphInterface,
        format: ExplanationFormat = ExplanationFormat.TEMPLATE,
    ) -> Explanation:
        title = f"Evidence Aggregation: {aggregation.conclusion_label}"
        summary = (
            f"Aggregated {aggregation.evidence_count} evidence paths converging on "
            f"{aggregation.conclusion_label} with confidence {aggregation.aggregated_confidence:.2f}."
        )
        exp_score = ExplainabilityScore.compute(
            confidence=aggregation.aggregated_confidence,
            evidence_count=aggregation.evidence_count,
            reasoning_depth=1,
        )
        raw_text = self._render_text(
            title, summary,
            [f"{aggregation.evidence_count} evidence sources, {aggregation.conflicting_evidence_count} conflicting"],
            f"{aggregation.evidence_count} supporting paths",
            f"Confidence {aggregation.aggregated_confidence:.2f} via {aggregation.aggregation_method}"
        )

        explanation_id = hashlib.sha256(
            f"explain:evidence:{aggregation.conclusion_node_id}:{format.value}".encode()
        ).hexdigest()

        return Explanation(
            explanation_id=explanation_id,
            inference_id="",
            format=format,
            title=title,
            summary=summary,
            steps=[f"{aggregation.evidence_count} evidence sources aggregated"],
            evidence_summary=f"{aggregation.evidence_count} supporting, {aggregation.conflicting_evidence_count} conflicting",
            confidence_explanation=f"Confidence {aggregation.aggregated_confidence:.2f} ({aggregation.aggregation_method})",
            explainability_score=exp_score,
            raw_text=raw_text,
        )
