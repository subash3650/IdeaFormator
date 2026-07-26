"""Base class for cluster providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipGraph


class ClusterProvider(ABC):
    """Abstract base class for all clustering providers.

    All implementations must inherit from this and register using
    the `register_provider` decorator.
    """

    @abstractmethod
    def cluster(self, graph: RelationshipGraph, config: ClusteringConfig) -> list[list[str]]:
        """Cluster the given relationship graph under configuration.

        Args:
            graph: The RelationshipGraph instance to cluster.
            config: The ClusteringConfig containing algorithm settings.

        Returns:
            A list of clusters, where each cluster is a list of member IDs.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Provider version identifier."""
