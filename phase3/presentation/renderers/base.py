from __future__ import annotations

from abc import ABC, abstractmethod

from phase3.presentation.config import PresentationConfig
from phase3.presentation.schema import PresentationModel, ReportFormat


class Renderer(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def format(self) -> ReportFormat:
        ...

    @abstractmethod
    def render(self, model: PresentationModel, config: PresentationConfig) -> str:
        ...
