"""Clustering providers package."""

from __future__ import annotations

# Import all providers to ensure they are registered
from phase2.clustering.providers.base import ClusterProvider
from phase2.clustering.providers.connected_components import ConnectedComponentsProvider
from phase2.clustering.providers.hierarchical import HierarchicalProvider
from phase2.clustering.providers.leiden import LeidenProvider
from phase2.clustering.providers.louvain import LouvainProvider
from phase2.clustering.providers.registry import (
    available_providers,
    create_provider,
    get_provider_class,
    register_provider,
)

__all__ = [
    "ClusterProvider",
    "ConnectedComponentsProvider",
    "HierarchicalProvider",
    "LeidenProvider",
    "LouvainProvider",
    "register_provider",
    "get_provider_class",
    "create_provider",
    "available_providers",
]
