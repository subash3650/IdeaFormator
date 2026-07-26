from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.schema import ChartSpec, ChartType


class ChartProvider(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def chart_type(self) -> ChartType:
        ...

    @property
    @abstractmethod
    def priority(self) -> int:
        ...

    @abstractmethod
    def build(self, data: dict[str, Any], config: PresentationConfig) -> ChartSpec:
        ...
