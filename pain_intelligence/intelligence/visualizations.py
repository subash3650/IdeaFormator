"""Minimal Plotly visualizations for the Knowledge Extraction Engine."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VisualizationFactory:
    """Generates publication-quality charts using Plotly.
    
    Minimal set for Phase 1.5. Extended in Phase 2+.
    """

    def __init__(self, theme: str = "plotly_dark", width: int = 1200, height: int = 600) -> None:
        self.theme = theme
        self.width = width
        self.height = height

    def _import_plotly(self):
        import plotly.graph_objects as go
        import plotly.express as px
        return go, px

    def platform_distribution(self, data: dict[str, int]) -> Any:
        go, px = self._import_plotly()
        fig = px.bar(
            x=list(data.keys()), y=list(data.values()),
            title="Platform Distribution",
            labels={"x": "Platform", "y": "Documents"},
            template=self.theme,
        )
        fig.update_layout(width=self.width, height=self.height)
        return fig

    def rating_distribution(self, data: dict[str, int]) -> Any:
        go, px = self._import_plotly()
        fig = px.bar(
            x=list(data.keys()), y=list(data.values()),
            title="Rating Distribution",
            labels={"x": "Rating", "y": "Count"},
            template=self.theme,
        )
        fig.update_layout(width=self.width, height=self.height)
        return fig

    def top_keywords(self, keywords: list[tuple[str, int]], title: str = "Top Keywords") -> Any:
        go, px = self._import_plotly()
        fig = px.bar(
            x=[k[0] for k in keywords],
            y=[k[1] for k in keywords],
            title=title,
            labels={"x": "Keyword", "y": "Frequency"},
            template=self.theme,
        )
        fig.update_layout(width=self.width, height=self.height)
        return fig

    def evidence_confidence(self, evidence_list: list[Any]) -> Any:
        go, px = self._import_plotly()
        names = [e.signal_key[:30] for e in evidence_list[:30]]
        confs = [e.confidence for e in evidence_list[:30]]
        fig = px.bar(
            x=names, y=confs,
            title="Evidence Confidence (Top 30)",
            labels={"x": "Signal", "y": "Confidence"},
            template=self.theme,
        )
        fig.update_layout(width=self.width, height=self.height, xaxis_tickangle=-45)
        return fig

    def save(self, fig: Any, path: str | Path, fmt: str = "html") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "html":
            fig.write_html(str(path))
        elif fmt == "png":
            fig.write_image(str(path))
        elif fmt == "svg":
            fig.write_image(str(path))