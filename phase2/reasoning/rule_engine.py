"""Rule engine — priority ordering, dependency resolution, rule application."""

from __future__ import annotations

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.rules.base import ReasoningRule
from phase2.reasoning.rules.registry import (
    available_rules,
    create_rule,
    get_rule_metadata,
    get_rules_sorted_by_priority,
)
from phase2.reasoning.schema import InferenceResult


class RuleEngine:
    def __init__(
        self,
        enabled_rules: list[str] | None = None,
        max_iterations: int = 5,
        max_inferences: int = 10000,
    ) -> None:
        self._enabled_rules = set(enabled_rules or [])
        self._max_iterations = max_iterations
        self._max_inferences = max_inferences
        self._rules: list[ReasoningRule] = []
        self._resolved: bool = False

    def initialize(self) -> None:
        self._rules = []
        sorted_rules = get_rules_sorted_by_priority()
        if self._enabled_rules:
            sorted_rules = [
                (n, c, m) for n, c, m in sorted_rules if n in self._enabled_rules
            ]
        for name, cls, meta in sorted_rules:
            rule = create_rule(name)
            self._rules.append(rule)
        self._resolve_dependencies()
        self._resolved = True

    def _resolve_dependencies(self) -> None:
        rule_map: dict[str, ReasoningRule] = {r.name: r for r in self._rules}
        enabled_names: set[str] = set(rule_map.keys())
        added = True
        while added:
            added = False
            for rule in list(self._rules):
                meta = get_rule_metadata(rule.name)
                for dep in meta.dependencies:
                    if dep in enabled_names:
                        continue
                    if dep in available_rules():
                        dep_rule = create_rule(dep)
                        self._rules.append(dep_rule)
                        rule_map[dep] = dep_rule
                        enabled_names.add(dep)
                        added = True
                    else:
                        if rule in self._rules:
                            self._rules.remove(rule)
                        break

    def initialize_rules(self, graph: GraphInterface) -> None:
        for rule in self._rules:
            rule.initialize(graph)

    @property
    def rules(self) -> list[ReasoningRule]:
        return list(self._rules)

    def match_rules(self, graph: GraphInterface) -> dict[str, list[str]]:
        matches: dict[str, list[str]] = {}
        all_nodes = [n.node_id for n in graph.nodes()]
        for rule in self._rules:
            matching: list[str] = []
            for nid in all_nodes:
                if rule.matches(graph, nid):
                    matching.append(nid)
            matches[rule.name] = matching
        return matches

    def apply_all(
        self,
        graph: GraphInterface,
        propagator: ConfidencePropagator,
    ) -> list[InferenceResult]:
        if not self._resolved:
            self.initialize()
        total_inferences: list[InferenceResult] = []
        for iteration in range(self._max_iterations):
            iteration_results: list[InferenceResult] = []
            for rule in self._rules:
                all_nodes = [n.node_id for n in graph.nodes()]
                for nid in all_nodes:
                    if len(total_inferences) + len(iteration_results) >= self._max_inferences:
                        break
                    if rule.matches(graph, nid):
                        try:
                            results = rule.apply(graph, nid, propagator)
                            iteration_results.extend(results)
                        except Exception:
                            continue
                if len(total_inferences) + len(iteration_results) >= self._max_inferences:
                    break
            if not iteration_results:
                break
            total_inferences.extend(iteration_results)
            if iteration >= 1 and len(iteration_results) < 10:
                break
        return total_inferences[: self._max_inferences]
