"""Leiden clustering provider stub (for future extensibility)."""

from __future__ import annotations

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipGraph
from phase2.clustering.providers.base import ClusterProvider
from phase2.clustering.providers.registry import register_provider


@register_provider("leiden")
class LeidenProvider(ClusterProvider):
    """Leiden community detection provider stub.

    The Leiden algorithm is an advanced community detection method that
    maximizes modularity or other objective functions. It guarantees connected
    communities and is highly scalable.

    Extension Points:
        1. Install python-igraph and leidenalg packages:
           `pip install igraph leidenalg`
        2. Create an igraph representation of RelationshipGraph:
           ```python
           import igraph as ig
           g = ig.Graph()
           # add nodes and edges
           ```
        3. Run Leiden algorithm:
           ```python
           import leidenalg
           partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, weights='weight')
           ```
        4. Return clusters as lists of member IDs.
    """

    def cluster(self, graph: RelationshipGraph, config: ClusteringConfig) -> list[list[str]]:
        """Leiden algorithm stub. Raises NotImplementedError."""
        raise NotImplementedError(
            "Leiden provider is a documented stub for future extensibility. "
            "To use Leiden, please follow the extension instructions in the "
            "LeidenProvider class documentation."
        )

    @property
    def name(self) -> str:
        return "leiden"

    @property
    def version(self) -> str:
        return "1.0-stub"
