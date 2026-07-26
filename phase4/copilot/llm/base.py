from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from phase4.copilot.schema import LLMProviderResponse


class BaseLLMProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def generate(self, prompt: str, context: dict[str, Any] | None = None) -> LLMProviderResponse:
        ...

    @abstractmethod
    def generate_stream(self, prompt: str, context: dict[str, Any] | None = None):
        ...

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        ...

    def supports_streaming(self) -> bool:
        return True
