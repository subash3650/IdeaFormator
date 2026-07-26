"""Abstract base class for reasoning rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.reasoning.confidence import ConfidencePropagator
from phase2.reasoning.schema import InferenceResult


@dataclass
class RuleMetadata:
    name: str
    version: str = "1.0"
    priority: int = 50
    dependencies: list[str] = field(default_factory=list)
    description: str = ""
    author: str = "system"


class ReasoningRule(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    def initialize(self, graph: GraphInterface) -> None:
        pass

    @abstractmethod
    def matches(self, graph: GraphInterface, node_id: str) -> bool:
        ...

    @abstractmethod
    def apply(
        self,
        graph: GraphInterface,
        node_id: str,
        propagator: ConfidencePropagator,
    ) -> list[InferenceResult]:
        ...
