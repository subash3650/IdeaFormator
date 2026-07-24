"""Abstract interface for embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class EmbeddingProvider(ABC):
    """Abstract embedding provider that all providers must implement."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Embed a batch of texts. Returns L2-normalized vectors."""

    @abstractmethod
    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Expected embedding dimension."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Canonical provider name (matches EmbeddingProviderType)."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Model name/path used for this provider."""

    @property
    @abstractmethod
    def model_fingerprint(self) -> str:
        """Short deterministic fingerprint of the model (version hash)."""