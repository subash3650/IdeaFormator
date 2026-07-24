"""Euclidean distance similarity provider."""

from __future__ import annotations

import numpy as np

from phase2.similarity.providers.base import SimilarityProvider
from phase2.similarity.providers.registry import register


@register("euclidean")
class EuclideanSimilarityProvider(SimilarityProvider):
    """Euclidean distance converted to similarity score in [0, 1]."""

    def compute_pairwise(self, matrix1: np.ndarray, matrix2: np.ndarray) -> np.ndarray:
        diff = matrix1[:, np.newaxis, :] - matrix2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        max_dist = np.sqrt(matrix1.shape[1] * 4.0)
        return 1.0 - distances / max_dist

    def compute_scores(self, query: np.ndarray, index: np.ndarray) -> np.ndarray:
        diff = index - query[np.newaxis, :]
        distances = np.sqrt(np.sum(diff**2, axis=-1))
        max_dist = np.sqrt(index.shape[1] * 4.0)
        return 1.0 - distances / max_dist

    @property
    def name(self) -> str:
        return "euclidean"

    @property
    def version(self) -> str:
        return "1.0"
