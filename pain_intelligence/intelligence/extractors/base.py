"""Base extractor interface for all observation-producing modules."""

from __future__ import annotations

from abc import ABC, abstractmethod

import polars as pl

from pain_intelligence.intelligence.schema import ObservationBundle


class Extractor(ABC):
    """Base class for all observation extractors.
    
    Every extractor reads a DataFrame and returns an ObservationBundle.
    """

    @abstractmethod
    def extract(self, df: pl.DataFrame) -> ObservationBundle:
        """Extract observations from a DataFrame."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this extractor."""
        ...