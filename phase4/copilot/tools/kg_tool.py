from __future__ import annotations

import time
from typing import Any

from phase4.copilot.tools.base import BaseTool
from phase4.copilot.tools.registry import register_tool
from phase4.copilot.schema import (
    Citation,
    CitationSource,
    Intent,
    PermissionType,
    ToolMetadata,
    ToolPriority,
    ToolResult,
)
from phase4.copilot.config import CopilotConfig


@register_tool("knowledge_graph")
class KnowledgeGraphTool(BaseTool):
    def __init__(self, config: CopilotConfig | None = None) -> None:
        self._config = config or CopilotConfig()

    @property
    def name(self) -> str:
        return "knowledge_graph"

    @property
    def metadata(self) -> ToolMetadata:
        return ToolMetadata(
            name=self.name,
            description="Query the knowledge graph for nodes, edges, relationships, and paths",
            supported_intents=[
                Intent.QUERY_KG,
                Intent.SEARCH,
                Intent.STATISTICS,
            ],
            priority=ToolPriority.HIGH,
            estimated_cost=2.0,
            estimated_latency_ms=200.0,
            permissions=PermissionType.READ_ONLY,
            cacheable=True,
        )

    def execute(self, params: dict[str, Any]) -> ToolResult:
        start = time.perf_counter()
        try:
            action = params.get("action", "search")
            query = params.get("query", "")
            node_type = params.get("node_type")
            node_id = params.get("node_id")
            top_k = params.get("top_k", 10)

            store = self._get_store()
            nodes = store.load_nodes()
            edges = store.load_edges()

            if action == "stats":
                data = self._stats(nodes, edges)
            elif action == "get_node":
                data = self._get_node(nodes, node_id or query)
            elif action == "neighbors":
                data = self._neighbors(nodes, edges, node_id or query)
            elif action == "by_type":
                data = self._by_type(nodes, node_type)
            elif action == "search":
                data = self._search(nodes, query, top_k)
            else:
                data = self._search(nodes, query, top_k)

            citations = self._build_citations(data, nodes)

            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=data,
                citations=citations,
                elapsed_ms=round(elapsed, 1),
            )
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000
            return ToolResult(
                tool_name=self.name,
                success=False,
                data={},
                elapsed_ms=round(elapsed, 1),
                error=str(e),
            )

    def _get_store(self) -> Any:
        from phase2.knowledge_graph.store import KnowledgeGraphStore
        return KnowledgeGraphStore(self._config.phase2_dir)

    def _stats(self, nodes: list, edges: list) -> dict[str, Any]:
        node_types: dict[str, int] = {}
        for n in nodes:
            nt = getattr(n, "node_type", "unknown")
            node_types[str(nt)] = node_types.get(str(nt), 0) + 1
        return {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "node_type_distribution": node_types,
        }

    def _get_node(self, nodes: list, node_id: str) -> dict[str, Any]:
        for n in nodes:
            if getattr(n, "node_id", "") == node_id:
                return self._serialize_node(n)
            if getattr(n, "label", "") == node_id:
                return self._serialize_node(n)
        return {"error": f"Node not found: {node_id}"}

    def _neighbors(self, nodes: list, edges: list, node_id: str) -> dict[str, Any]:
        found = self._get_node(nodes, node_id)
        if "error" in found:
            return found

        neighbor_ids: set[str] = set()
        for e in edges:
            if getattr(e, "source_node_id", "") == node_id:
                neighbor_ids.add(getattr(e, "target_node_id", ""))
            if getattr(e, "target_node_id", "") == node_id:
                neighbor_ids.add(getattr(e, "source_node_id", ""))

        neighbors = [self._serialize_node(n) for n in nodes if getattr(n, "node_id", "") in neighbor_ids]
        return {"node": found, "neighbors": neighbors, "neighbor_count": len(neighbors)}

    def _by_type(self, nodes: list, node_type: str | None) -> dict[str, Any]:
        if not node_type:
            return {"error": "node_type required"}
        filtered = [self._serialize_node(n) for n in nodes if str(getattr(n, "node_type", "")).lower() == node_type.lower()]
        return {"node_type": node_type, "count": len(filtered), "nodes": filtered[:20]}

    def _search(self, nodes: list, query: str, top_k: int) -> dict[str, Any]:
        q = query.lower()
        scored = []
        for n in nodes:
            label = (getattr(n, "label", "") or "").lower()
            nid = (getattr(n, "node_id", "") or "").lower()
            ntype = (str(getattr(n, "node_type", ""))).lower()
            props = str(getattr(n, "properties", {})).lower()

            score = 0
            if q in label:
                score += 3
            if q in nid:
                score += 2
            if q in ntype:
                score += 1
            if q in props:
                score += 1

            if score > 0:
                scored.append((score, self._serialize_node(n)))

        scored.sort(key=lambda x: x[0], reverse=True)
        return {
            "query": query,
            "total": len(scored),
            "results": [s[1] for s in scored[:top_k]],
        }

    def _serialize_node(self, n: Any) -> dict[str, Any]:
        return {
            "node_id": getattr(n, "node_id", ""),
            "node_type": str(getattr(n, "node_type", "")),
            "label": getattr(n, "label", ""),
            "confidence": getattr(n, "confidence", 0.0),
            "properties": getattr(n, "properties", {}),
        }

    def _build_citations(self, data: dict, nodes: list) -> list[Citation]:
        citations: list[Citation] = []
        seen: set[str] = set()
        for result in data.get("results", []):
            nid = result.get("node_id", "")
            if nid and nid not in seen:
                seen.add(nid)
                citations.append(Citation(
                    source_module=CitationSource.KNOWLEDGE_GRAPH,
                    source_id=nid,
                    source_title=result.get("label", ""),
                    confidence=result.get("confidence", 0.0),
                    snippet=f"{result.get('node_type', '')}: {result.get('label', '')}",
                ))
        node = data.get("node")
        if node and node.get("node_id", "") not in seen:
            citations.append(Citation(
                source_module=CitationSource.KNOWLEDGE_GRAPH,
                source_id=node["node_id"],
                source_title=node.get("label", ""),
                confidence=node.get("confidence", 0.0),
                snippet=f"{node.get('node_type', '')}: {node.get('label', '')}",
            ))
        return citations
