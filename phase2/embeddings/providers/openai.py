"""OpenAI embedding provider stub."""

from __future__ import annotations

import numpy as np

from phase2.embeddings.config import EmbeddingEngineConfig
from phase2.embeddings.providers.base import EmbeddingProvider
from phase2.embeddings.registry import register


@register("openai")
class OpenAIProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingEngineConfig) -> None:
        self._config = config
        self._model_name = config.model

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        raise NotImplementedError("OpenAI provider is not yet implemented")

    def embed_one(self, text: str) -> np.ndarray:
        raise NotImplementedError("OpenAI provider is not yet implemented")

    @property
    def dimension(self) -> int:
        return self._config.dimension

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_fingerprint(self) -> str:
        return "stub"