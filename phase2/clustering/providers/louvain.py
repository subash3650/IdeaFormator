"""Louvain clustering provider stub (for future extensibility)."""

from __future__ import annotations

from phase2.clustering.config import ClusteringConfig
from phase2.clustering.graph import RelationshipGraph
from phase2.clustering.providers.base import ClusterProvider
from phase2.clustering.providers.registry import register_provider


@register_provider("louvain")
class LouvainProvider(ClusterProvider):
    """Louvain community detection provider stub.

    The Louvain algorithm is a fast multi-level algorithm for community
    detection that maximizes modularity.

    Extension Points:
        1. Install community / python-louvain packages:
           `pip install python-louvain`
        2. NetworkX/igraph adapter can be used, or a custom louvain implementation:
           ```python
           import community as community_louvain
           # Convert RelationshipGraph to networkx graph g
           partition = community_louvain.best_partition(g, weight='weight')
           ```
        3. Convert partition output (node -> community_id) back to lists of nodes.
        4. Return clusters as lists of member IDs.
    """

    def cluster(self, graph: RelationshipGraph, config: ClusteringConfig) -> list[list[str]]:
        """Louvain algorithm stub. Raises NotImplementedError."""
        raise NotImplementedError(
            "Louvain provider is a documented stub for future extensibility. "
            "To use Louvain, please follow the extension instructions in the "
            "LouvainProvider class documentation."
        )

    @property
    def name(self) -> str:
        return "louvain"

    @property
    def version(self) -> str:
        return "1.0-stub"
