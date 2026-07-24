"""Abstract interface for similarity providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SimilarityProvider(ABC):
    """Abstract similarity provider that all providers must implement."""

    @abstractmethod
    def compute_pairwise(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        """Compute pairwise similarity between two matrices.

        Args:
            matrix1: Shape [n1, d]
            matrix2: Shape [n2, d]

        Returns:
            Shape [n1, n2] similarity matrix.
        """

    @abstractmethod
    def compute_scores(self, query: np.ndarray, index: np.ndarray) -> np.ndarray:
        """Compute similarity scores for a single query against an index.

        Args:
            query: Shape [d]
            index: Shape [n, d]

        Returns:
            Shape [n] similarity scores.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Canonical provider name."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Provider version identifier."""
