"""GraphExporter — multi-format export for knowledge graph."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from phase2.knowledge_graph.evaluator import GraphEvaluation, GraphEvaluator
from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.store import KnowledgeGraphStore


class GraphExporter:
    """Exports knowledge graph in multiple formats."""

    def __init__(self, graph: GraphInterface, store: KnowledgeGraphStore, evaluator: GraphEvaluator | None = None) -> None:
        self._graph = graph
        self._store = store
        self._evaluator = evaluator or GraphEvaluator()

    def export_gexf(self, path: Path) -> Path:
        """Export graph as GEXF XML for Gephi."""
        lines: list[str] = []
        lines.append('<?xml version="1.0" encoding="UTF-8"?>')
        lines.append('<gexf xmlns="http://www.gexf.net/1.2draft" version="1.2">')
        lines.append("  <graph mode=\"static\" defaultedgetype=\"directed\">")

        # Nodes
        lines.append(f"    <nodes count=\"{self._graph.node_count()}\">")
        for node in self._graph.nodes():
            label_esc = node.label.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\"", "&quot;")
            lines.append(f'      <node id="{node.node_id}" label="{label_esc}" />')
        lines.append("    </nodes>")

        # Edges
        lines.append(f"    <edges count=\"{self._graph.edge_count()}\">")
        for i, edge in enumerate(self._graph.edges()):
            lines.append(
                f'      <edge id="{i}" source="{edge.source_node_id}" '
                f'target="{edge.target_node_id}" weight="{edge.weight}" '
                f'label="{edge.edge_type.value}" />'
            )
        lines.append("    </edges>")

        lines.append("  </graph>")
        lines.append("</gexf>")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def export_json(self, path: Path) -> Path:
        """Export graph as JSON adjacency list."""
        data: dict[str, Any] = {
            "graph_id": "",
            "node_count": self._graph.node_count(),
            "edge_count": self._graph.edge_count(),
            "nodes": [],
            "edges": [],
        }
        for node in self._graph.nodes():
            data["nodes"].append({
                "node_id": node.node_id,
                "node_type": node.node_type.value,
                "label": node.label,
                "confidence": node.confidence,
            })
        for edge in self._graph.edges():
            data["edges"].append({
                "edge_id": edge.edge_id,
                "source_node_id": edge.source_node_id,
                "target_node_id": edge.target_node_id,
                "edge_type": edge.edge_type.value,
                "weight": edge.weight,
                "confidence": edge.confidence,
            })
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path

    def export_csv(self, path: Path) -> dict[str, Path]:
        """Export graph as CSV files (nodes.csv + edges.csv)."""
        nodes_path = path / "nodes.csv" if path.suffix == "" else path.parent / "nodes.csv"
        edges_path = path / "edges.csv" if path.suffix == "" else path.parent / "edges.csv"

        nodes_path.parent.mkdir(parents=True, exist_ok=True)

        # Nodes CSV
        with open(nodes_path, "w", encoding="utf-8") as f:
            f.write("node_id,node_type,label,confidence\n")
            for node in self._graph.nodes():
                label_esc = node.label.replace('"', '""')
                f.write(f"{node.node_id},{node.node_type.value},\"{label_esc}\",{node.confidence}\n")

        # Edges CSV
        with open(edges_path, "w", encoding="utf-8") as f:
            f.write("edge_id,source_node_id,target_node_id,edge_type,weight,confidence\n")
            for edge in self._graph.edges():
                f.write(f"{edge.edge_id},{edge.source_node_id},{edge.target_node_id},{edge.edge_type.value},{edge.weight},{edge.confidence}\n")

        return {"nodes": nodes_path, "edges": edges_path}

    def export_statistics(self, path: Path) -> Path:
        """Export degree distribution and type statistics as JSON."""
        degree_dist: dict[str, int] = {}
        for node in self._graph.nodes():
            deg = self._graph.degree(node.node_id)
            degree_dist[str(deg)] = degree_dist.get(str(deg), 0) + 1

        type_dist: dict[str, int] = {}
        for node in self._graph.nodes():
            type_dist[node.node_type.value] = type_dist.get(node.node_type.value, 0) + 1

        edge_type_dist: dict[str, int] = {}
        for edge in self._graph.edges():
            edge_type_dist[edge.edge_type.value] = edge_type_dist.get(edge.edge_type.value, 0) + 1

        stats: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "node_count": self._graph.node_count(),
            "edge_count": self._graph.edge_count(),
            "degree_distribution": degree_dist,
            "node_type_distribution": type_dist,
            "edge_type_distribution": edge_type_dist,
            "avg_degree": round(sum(self._graph.degree(n.node_id) for n in self._graph.nodes()) / max(self._graph.node_count(), 1), 6),
        }
        path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
        return path

    def export_dashboard(self, path: Path) -> dict[str, Path]:
        """Export machine-readable dashboard JSON and human-readable TXT."""
        evaluation = self._evaluator.evaluate(self._graph)
        dash_json: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "node_count": evaluation["node_count"],
            "edge_count": evaluation["edge_count"],
            "density": evaluation["density"],
            "connected_components": evaluation["connected_components"],
            "largest_component_ratio": evaluation["largest_component_ratio"],
            "avg_confidence": evaluation["avg_confidence"],
            "avg_degree": evaluation["avg_degree"],
            "health": dict(evaluation["health"]),
            "warnings": evaluation["warnings"],
            "recommendations": evaluation["recommendations"],
            "type_distribution": evaluation["type_distribution"],
        }
        # Write JSON
        json_path = path.with_suffix(".json") if path.suffix == "" else path.parent / "graph_dashboard.json"
        json_path.write_text(json.dumps(dash_json, indent=2, default=str), encoding="utf-8")

        # Write TXT
        txt_path = path.with_suffix(".txt") if path.suffix == "" else path.parent / "graph_dashboard.txt"
        lines: list[str] = [
            "=" * 60,
            "  KNOWLEDGE GRAPH DASHBOARD",
            "=" * 60,
            f"  Generated: {dash_json['generated_at']}",
            "",
            "  OVERVIEW",
            f"    Nodes:               {dash_json['node_count']}",
            f"    Edges:               {dash_json['edge_count']}",
            f"    Density:             {dash_json['density']:.6f}",
            f"    Connected Components: {dash_json['connected_components']}",
            f"    Largest Component:   {dash_json['largest_component_ratio']:.1%}",
            "",
            "  QUALITY",
            f"    Health Score:        {dash_json['health']['score']}/100 ({dash_json['health']['status']})",
            f"    Avg Confidence:      {dash_json['avg_confidence']:.4f}",
            f"    Avg Degree:          {dash_json['avg_degree']:.4f}",
            "",
        ]
        if dash_json["warnings"]:
            lines.append("  WARNINGS")
            for w in dash_json["warnings"]:
                lines.append(f"    - {w}")
            lines.append("")
        if dash_json["recommendations"]:
            lines.append("  RECOMMENDATIONS")
            for r in dash_json["recommendations"]:
                lines.append(f"    - {r}")
            lines.append("")

        lines.append("  NODE TYPE DISTRIBUTION")
        for t, c in sorted(dash_json["type_distribution"].items()):
            lines.append(f"    {t:20s}: {c}")
        lines.append("")
        lines.append("=" * 60)

        txt_path.write_text("\n".join(lines), encoding="utf-8")
        return {"json": json_path, "txt": txt_path}

    def export_summary(self, path: Path) -> Path:
        """Export human-readable Markdown summary."""
        evaluation = self._evaluator.evaluate(self._graph)
        lines: list[str] = [
            "# Knowledge Graph Summary",
            "",
            f"**Generated:** {datetime.now(timezone.utc).isoformat()}",
            f"**Nodes:** {evaluation['node_count']}",
            f"**Edges:** {evaluation['edge_count']}",
            f"**Health Score:** {evaluation['health']['score']}/100 ({evaluation['health']['status']})",
            "",
            "## Node Distribution",
            "| Type | Count |",
            "|---|---|",
        ]
        for t, c in sorted(evaluation["type_distribution"].items()):
            lines.append(f"| {t} | {c} |")
        lines.append("")
        lines.append("## Edge Distribution")
        lines.append("| Type | Count |")
        lines.append("|---|---|")
        for t, c in sorted(evaluation["edge_type_distribution"].items()):
            lines.append(f"| {t} | {c} |")
        lines.append("")
        lines.append("## Quality Metrics")
        lines.append(f"- **Density:** {evaluation['density']:.6f}")
        lines.append(f"- **Connected Components:** {evaluation['connected_components']}")
        lines.append(f"- **Largest Component Ratio:** {evaluation['largest_component_ratio']:.1%}")
        lines.append(f"- **Avg Confidence:** {evaluation['avg_confidence']:.4f}")
        lines.append(f"- **Avg Degree:** {evaluation['avg_degree']:.4f}")
        lines.append(f"- **Orphan Nodes:** {evaluation['orphan_node_count']}")
        lines.append(f"- **Orphan Edges:** {evaluation['orphan_edge_count']}")
        if evaluation["warnings"]:
            lines.append("")
            lines.append("## Warnings")
            for w in evaluation["warnings"]:
                lines.append(f"- {w}")
        if evaluation["recommendations"]:
            lines.append("")
            lines.append("## Recommendations")
            for r in evaluation["recommendations"]:
                lines.append(f"- {r}")
        lines.append("")
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def export_report(self, path: Path) -> Path:
        """Export full evaluation report as JSON."""
        evaluation = self._evaluator.evaluate(self._graph)
        report: dict[str, Any] = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "evaluation": dict(evaluation["health"]),
            "node_count": evaluation["node_count"],
            "edge_count": evaluation["edge_count"],
            "density": evaluation["density"],
            "connected_components": evaluation["connected_components"],
            "largest_component_ratio": evaluation["largest_component_ratio"],
            "avg_confidence": evaluation["avg_confidence"],
            "avg_degree": evaluation["avg_degree"],
            "orphan_node_count": evaluation["orphan_node_count"],
            "orphan_edge_count": evaluation["orphan_edge_count"],
            "type_distribution": evaluation["type_distribution"],
            "edge_type_distribution": evaluation["edge_type_distribution"],
            "warnings": evaluation["warnings"],
            "recommendations": evaluation["recommendations"],
        }
        path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        return path
