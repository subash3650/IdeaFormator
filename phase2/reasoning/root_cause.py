"""Root cause discovery — find root causes using transitive impact ranking."""

from __future__ import annotations

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType, NodeType
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.schema import RootCause, RootCauseRanking


class RootCauseDiscoverer:
    def __init__(
        self,
        ranking: RootCauseRanking = RootCauseRanking.TRANSITIVE_IMPACT,
        max_depth: int = 8,
        min_confidence: float = 0.15,
    ) -> None:
        self._ranking = ranking
        self._max_depth = max_depth
        self._min_confidence = min_confidence

    def discover(
        self,
        graph: GraphInterface,
        effect_node_ids: list[str] | None = None,
        propagator: ConfidencePropagator | None = None,
    ) -> list[RootCause]:
        if effect_node_ids is None:
            effect_node_ids = [
                n.node_id for n in graph.nodes()
                if n.node_type == NodeType.PROBLEM_SIGNAL
            ]

        all_causes: list[RootCause] = []
        for effect_id in effect_node_ids:
            causes = self._trace_ancestors(graph, effect_id, propagator)
            all_causes.extend(causes)

        self._compute_transitive_impact(graph, all_causes)
        all_causes = self._rank(all_causes)
        return all_causes

    def _trace_ancestors(
        self,
        graph: GraphInterface,
        node_id: str,
        propagator: ConfidencePropagator | None = None,
    ) -> list[RootCause]:
        results: list[RootCause] = []
        effect = graph.get_node(node_id)
        if effect is None:
            return results
        effect_label = effect.label or node_id

        def dfs(current: str, path: list[str], depth: int) -> None:
            if depth > self._max_depth:
                return
            predecessors = graph.predecessors(current, edge_type=EdgeType.CAUSES)
            predecessors.extend(
                graph.predecessors(current, edge_type=EdgeType.BLOCKS)
            )
            predecessors.extend(
                graph.predecessors(current, edge_type=EdgeType.DEPENDS_ON)
            )
            predecessors = list(dict.fromkeys(predecessors))

            if not predecessors:
                if len(path) >= 2:
                    self._add_cause(graph, path, node_id, effect_label, results, propagator)
                return

            for pred in predecessors:
                if pred in path:
                    continue
                path.append(pred)
                dfs(pred, path, depth + 1)
                path.pop()

        dfs(node_id, [node_id], 0)

        if not results:
            self._add_cause(graph, [node_id], node_id, effect_label, results, propagator)

        return results

    def _add_cause(
        self,
        graph: GraphInterface,
        path: list[str],
        effect_id: str,
        effect_label: str,
        results: list[RootCause],
        propagator: ConfidencePropagator | None = None,
    ) -> None:
        cause_id = path[-1]
        cause = graph.get_node(cause_id)
        if cause is None:
            return
        cause_label = cause.label or cause_id

        conf = 0.0
        if propagator:
            conf = propagator.propagate(path, graph)
        elif len(path) >= 2:
            conf = 0.5 ** (len(path) - 1)

        existing = [r for r in results if r.cause_node_id == cause_id and r.effect_node_id == effect_id]
        if existing:
            if conf > existing[0].propagated_confidence:
                existing[0] = existing[0].model_copy(update={
                    "propagated_confidence": round(conf, 4),
                    "path": path,
                    "path_length": len(path) - 1,
                })
            return

        results.append(RootCause(
            cause_node_id=cause_id,
            cause_label=cause_label,
            effect_node_id=effect_id,
            effect_label=effect_label,
            path=path,
            path_length=len(path) - 1,
            propagated_confidence=round(conf, 4),
            transitive_impact_count=0,
            evidence_count=0,
            ranking_score=0.0,
            ranking_method=self._ranking,
        ))

    def _compute_transitive_impact(
        self,
        graph: GraphInterface,
        causes: list[RootCause],
    ) -> None:
        for rc in causes:
            count = self._count_downstream_effects(graph, rc.cause_node_id)
            evidence_count = len(graph.predecessors(rc.effect_node_id, edge_type=EdgeType.REFERENCES))
            evidence_count += len(graph.predecessors(rc.effect_node_id, edge_type=EdgeType.SUPPORTED_BY))
            idx = causes.index(rc)
            causes[idx] = rc.model_copy(update={
                "transitive_impact_count": count,
                "evidence_count": evidence_count,
            })

    def _count_downstream_effects(self, graph: GraphInterface, node_id: str) -> int:
        visited: set[str] = set()
        stack = [node_id]
        count = 0
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            if current != node_id:
                count += 1
            for succ in graph.successors(current, edge_type=EdgeType.CAUSES):
                if succ not in visited:
                    stack.append(succ)
            for succ in graph.successors(current, edge_type=EdgeType.BLOCKS):
                if succ not in visited:
                    stack.append(succ)
            for succ in graph.successors(current, edge_type=EdgeType.DEPENDS_ON):
                if succ not in visited:
                    stack.append(succ)
        return max(0, count)

    def _rank(self, causes: list[RootCause]) -> list[RootCause]:
        if self._ranking == RootCauseRanking.TRANSITIVE_IMPACT:
            max_impact = max((c.transitive_impact_count for c in causes), default=1)
            max_evidence = max((c.evidence_count for c in causes), default=1)
            if max_evidence == 0:
                max_evidence = 1
            ranked = []
            for rc in causes:
                impact_norm = rc.transitive_impact_count / max(1, max_impact)
                evidence_norm = rc.evidence_count / max(1, max_evidence)
                score = impact_norm * rc.propagated_confidence * (evidence_norm + 0.5)
                ranked.append(rc.model_copy(update={
                    "ranking_score": round(score, 4),
                }))
            ranked.sort(key=lambda x: x.ranking_score, reverse=True)
            return ranked

        if self._ranking == RootCauseRanking.CONFIDENCE:
            ranked = sorted(causes, key=lambda x: x.propagated_confidence, reverse=True)
            return [
                rc.model_copy(update={"ranking_score": rc.propagated_confidence})
                for rc in ranked
            ]

        if self._ranking == RootCauseRanking.DEPTH:
            ranked = sorted(causes, key=lambda x: x.path_length, reverse=True)
            max_depth = max((c.path_length for c in ranked), default=1)
            return [
                rc.model_copy(update={"ranking_score": round(rc.path_length / max_depth, 4)})
                for rc in ranked
            ]

        return causes
