"""Sentence-Transformers provider backed by huggingface models."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from phase2.embeddings.config import EmbeddingEngineConfig
from phase2.embeddings.providers.base import EmbeddingProvider
from phase2.embeddings.registry import register


@register("sentence_transformers")
class SentenceTransformerProvider(EmbeddingProvider):
    def __init__(self, config: EmbeddingEngineConfig) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = config.model
        self._device = self._resolve_device(config.device)
        self._batch_size = config.batch_size
        self._normalize = config.normalize
        self._config = config
        self._model: SentenceTransformer = SentenceTransformer(
            self._model_name,
            device=self._device,
        )
        self._fingerprint: str | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return "cpu"
        return device

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def embed_one(self, text: str) -> np.ndarray:
        vec = self._model.encode(
            text,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )
        return np.array(vec, dtype=np.float32)

    @property
    def dimension(self) -> int:
        return self._model.get_embedding_dimension() or self._config.dimension

    @property
    def provider_name(self) -> str:
        return "sentence_transformers"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_fingerprint(self) -> str:
        if self._fingerprint is not None:
            return self._fingerprint
        try:
            import hashlib

            if hasattr(self._model, "_model_config") and self._model._model_config:
                cfg_str = str(sorted(self._model._model_config.items()))
                self._fingerprint = hashlib.sha256(cfg_str.encode()).hexdigest()[:12]
            else:
                module_path = Path(self._model._modules_path) if hasattr(self._model, "_modules_path") else None
                if module_path and module_path.exists():
                    digests = []
                    for fpath in sorted(module_path.rglob("*")):
                        if fpath.is_file() and fpath.suffix in {".bin", ".safetensors", ".json", ".py"}:
                            digests.append(hashlib.sha256(fpath.read_bytes()).hexdigest()[:8])
                    self._fingerprint = hashlib.sha256("".join(digests).encode()).hexdigest()[:12] if digests else "unknown"
                else:
                    self._fingerprint = hashlib.sha256(self._model_name.encode()).hexdigest()[:12]
        except Exception:
            import hashlib
            self._fingerprint = hashlib.sha256(self._model_name.encode()).hexdigest()[:12]
        return self._fingerprint