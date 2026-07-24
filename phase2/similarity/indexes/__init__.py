"""Vector index implementations for nearest-neighbour search."""

from phase2.similarity.indexes.base import VectorIndex
from phase2.similarity.indexes.linear import LinearIndex

__all__ = ["VectorIndex", "LinearIndex"]
