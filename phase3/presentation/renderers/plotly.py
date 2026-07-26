from __future__ import annotations

import json
from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers.base import Renderer
from phase3.presentation.renderers.registry import register_renderer
from phase3.presentation.schema import ChartSpec, ChartType, PresentationModel, ReportFormat


@register_renderer(name="plotly")
class PlotlyRenderer(Renderer):
    @property
    def name(self) -> str:
        return "plotly"

    @property
    def format(self) -> ReportFormat:
        return ReportFormat.json

    def render(self, model: PresentationModel, config: PresentationConfig) -> str:
        all_charts: dict[str, dict[str, Any]] = {}
        for section in model.sections:
            for chart in section.charts:
                fig = self.chart_to_plotly_json(chart, config)
                if fig:
                    all_charts[chart.chart_id] = fig

        for chart in model.assets.charts:
            fig = self.chart_to_plotly_json(chart, config)
            if fig:
                all_charts[chart.chart_id] = fig

        return json.dumps({"charts": all_charts}, indent=2, default=str)

    def chart_to_plotly_json(self, spec: ChartSpec, config: PresentationConfig) -> dict[str, Any] | None:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        fig = self._build_figure(spec, config)
        if fig is None:
            return None

        fig.update_layout(
            title=spec.title,
            width=config.chart_width,
            height=config.chart_height,
            template=config.chart_theme,
        )

        return fig.to_dict()

    def chart_to_html(self, spec: ChartSpec, config: PresentationConfig) -> str:
        try:
            import plotly.io as pio
        except ImportError:
            return f"<div class='chart-placeholder'>Chart: {spec.title} ({spec.chart_type.value})</div>"

        fig = self._build_figure(spec, config)
        if fig is None:
            return f"<div class='chart-placeholder'>Chart: {spec.title} ({spec.chart_type.value})</div>"

        fig.update_layout(
            title=spec.title,
            width=config.chart_width,
            height=config.chart_height,
            template=config.chart_theme,
        )

        return pio.to_html(fig, full_html=False, include_plotlyjs=False)

    def _build_figure(self, spec: ChartSpec, config: PresentationConfig) -> Any | None:
        try:
            import plotly.graph_objects as go
        except ImportError:
            return None

        dispatch: dict[ChartType, Any] = {
            ChartType.bar: self._build_bar,
            ChartType.line: self._build_line,
            ChartType.pie: self._build_pie,
            ChartType.timeline: self._build_timeline,
            ChartType.heatmap: self._build_heatmap,
            ChartType.treemap: self._build_treemap,
            ChartType.sankey: self._build_sankey,
        }

        builder = dispatch.get(spec.chart_type)
        if builder is None:
            return None

        return builder(spec, config)

    def _build_bar(self, spec: ChartSpec, config: PresentationConfig) -> Any:
        import plotly.graph_objects as go

        orientation = spec.metadata.get("orientation", "v")
        stacked = spec.metadata.get("stacked", False)
        fig = go.Figure()

        for series in spec.series:
            kwargs: dict[str, Any] = {"name": series.name}
            if orientation == "h":
                kwargs["y"] = spec.labels
                kwargs["x"] = series.values
            else:
                kwargs["x"] = spec.labels
                kwargs["y"] = series.values

            if series.color:
                kwargs["marker_color"] = series.color

            fig.add_trace(go.Bar(**kwargs))

        if stacked:
            fig.update_layout(barmode="stack")

        return fig

    def _build_line(self, spec: ChartSpec, config: PresentationConfig) -> Any:
        import plotly.graph_objects as go

        show_markers = spec.metadata.get("show_markers", True)
        smooth = spec.metadata.get("smooth", False)
        fig = go.Figure()

        for series in spec.series:
            kwargs: dict[str, Any] = {
                "name": series.name,
                "x": spec.labels,
                "y": series.values,
                "mode": "lines+markers" if show_markers else "lines",
            }
            if series.color:
                kwargs["line"] = {"color": series.color}
            if smooth:
                kwargs["line"] = {**kwargs.get("line", {}), "shape": "spline"}

            fig.add_trace(go.Scatter(**kwargs))

        return fig

    def _build_pie(self, spec: ChartSpec, config: PresentationConfig) -> Any:
        import plotly.graph_objects as go

        hole = spec.metadata.get("hole", 0.0)
        series = spec.series[0] if spec.series else None
        values = series.values if series else []
        labels = spec.labels

        return go.Figure(
            data=[
                go.Pie(labels=labels, values=values, hole=hole, showlegend=True),
            ]
        )

    def _build_timeline(self, spec: ChartSpec, config: PresentationConfig) -> Any:
        import plotly.graph_objects as go

        fig = go.Figure()

        for i, series in enumerate(spec.series):
            label = spec.labels[i] if i < len(spec.labels) else f"Item {i}"
            start = series.values[0] if len(series.values) > 0 else ""
            end = series.values[1] if len(series.values) > 1 else ""

            kwargs: dict[str, Any] = {
                "name": label,
                "x": [end] if end else [start],
                "y": [label],
                "orientation": "h",
                "base": [start] if start else None,
            }
            if series.color:
                kwargs["marker_color"] = series.color

            fig.add_trace(go.Bar(**kwargs))

        fig.update_layout(barmode="overlay")

        return fig

    def _build_heatmap(self, spec: ChartSpec, config: PresentationConfig) -> Any:
        import plotly.graph_objects as go

        z: list[list[float]] = []
        y_labels: list[str] = []

        for series in spec.series:
            row = [float(v) for v in series.values]
            z.append(row)
            y_labels.append(series.name)

        x_labels = spec.labels
        colorscale = spec.metadata.get("colorscale", "Viridis")

        return go.Figure(
            data=[
                go.Heatmap(z=z, x=x_labels, y=y_labels, colorscale=colorscale),
            ]
        )

    def _build_treemap(self, spec: ChartSpec, config: PresentationConfig) -> Any:
        import plotly.graph_objects as go

        labels = spec.labels
        parents = spec.metadata.get("parents", [])
        values = spec.series[0].values if spec.series else []
        branch_values = spec.metadata.get("branch_values", "total")

        return go.Figure(
            data=[
                go.Treemap(
                    labels=labels,
                    parents=parents,
                    values=values,
                    branchvalues=branch_values,
                ),
            ]
        )

    def _build_sankey(self, spec: ChartSpec, config: PresentationConfig) -> Any:
        import plotly.graph_objects as go

        labels = spec.labels
        source_str: list[str] = spec.series[0].values if len(spec.series) > 0 else []
        target_str: list[str] = spec.series[1].values if len(spec.series) > 1 else []
        value: list[float] = [float(v) for v in spec.series[2].values] if len(spec.series) > 2 else []

        label_to_idx = {label: i for i, label in enumerate(labels)}
        source_idx = [label_to_idx.get(s, 0) for s in source_str]
        target_idx = [label_to_idx.get(t, 0) for t in target_str]
        arrangement = spec.metadata.get("arrangement", "snap")

        return go.Figure(
            data=[
                go.Sankey(
                    arrangement=arrangement,
                    node={"label": labels},
                    link={"source": source_idx, "target": target_idx, "value": value},
                ),
            ]
        )
