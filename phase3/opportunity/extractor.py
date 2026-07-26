"""OpportunityExtractor — merges multiple intelligence sources into candidates."""

from __future__ import annotations

import hashlib
from typing import Any

from phase3.opportunity.config import OpportunityConfig
from phase3.opportunity.schema import MarketSize


class OpportunityExtractor:
    """Extracts opportunity candidates by merging root causes, evidence,
    reasoning chains, knowledge graph nodes, and semantic clusters."""

    def __init__(self, config: OpportunityConfig) -> None:
        self._config = config

    def extract(
        self,
        root_causes: list,
        evidence_aggregations: list,
        inferences: list,
        chains: list,
        kg_nodes: list,
        kg_edges: list,
        clusters: list,
    ) -> list[dict]:
        """Produce raw opportunity candidates from upstream data.

        Merges intelligence from:
          - Root Causes (primary signal)
          - Evidence Aggregations (supporting evidence)
          - Reasoning Chains (reasoning provenance)
          - Knowledge Graph nodes/edges (entity extraction)
          - Semantic Clusters (group density)

        Returns a list of candidate dicts ready for scoring.
        """
        candidates: list[dict] = []

        # Build lookup indices
        evidence_by_target: dict[str, list] = {}
        for ea in evidence_aggregations:
            eid = ea.conclusion_node_id if hasattr(ea, "conclusion_node_id") else ""
            evidence_by_target.setdefault(eid, []).append(ea)

        cluster_membership: dict[str, list] = {}
        for cl in clusters:
            mids = cl.member_ids if hasattr(cl, "member_ids") else []
            for mid in mids:
                cluster_membership.setdefault(mid, []).append(cl)

        kg_node_map: dict[str, Any] = {}
        products: set[str] = set()
        companies: set[str] = set()
        technologies: set[str] = set()
        for n in kg_nodes:
            nid = n.node_id if hasattr(n, "node_id") else ""
            kg_node_map[nid] = n
            nt = n.node_type.value if hasattr(n, "node_type") and hasattr(n.node_type, "value") else ""
            label = n.label if hasattr(n, "label") else ""
            if nt == "product":
                products.add(nid)
                products.add(label.lower())
            elif nt == "company":
                companies.add(nid)
                companies.add(label.lower())
            elif nt == "technology":
                technologies.add(nid)
                technologies.add(label.lower())

        # Build edge-based adjacency for entity lookups
        node_edges: dict[str, list] = {}
        for e in kg_edges:
            src = e.source_node_id if hasattr(e, "source_node_id") else ""
            tgt = e.target_node_id if hasattr(e, "target_node_id") else ""
            node_edges.setdefault(src, []).append(e)
            node_edges.setdefault(tgt, []).append(e)

        # 1. Candidates from root causes
        for rc in root_causes:
            cause_id = rc.cause_node_id if hasattr(rc, "cause_node_id") else ""
            effect_id = rc.effect_node_id if hasattr(rc, "effect_node_id") else ""
            cause_label = rc.cause_label if hasattr(rc, "cause_label") else cause_id
            effect_label = rc.effect_label if hasattr(rc, "effect_label") else effect_id

            cand = self._make_candidate(
                root_problem=cause_id,
                title=cause_label,
                summary=f"Root cause '{cause_label}' affects '{effect_label}'",
                rc=rc,
                evidence_by_target=evidence_by_target,
                cluster_membership=cluster_membership,
                kg_node_map=kg_node_map,
                node_edges=node_edges,
                products=products,
                companies=companies,
                technologies=technologies,
            )
            candidates.append(cand)

        # 2. Candidates from evidence aggregations not already covered
        covered_problems = {c["root_problem"] for c in candidates}
        for ea in evidence_aggregations:
            eid = ea.conclusion_node_id if hasattr(ea, "conclusion_node_id") else ""
            if eid in covered_problems:
                continue
            label = ea.conclusion_label if hasattr(ea, "conclusion_label") else eid
            cand = self._make_candidate(
                root_problem=eid,
                title=label,
                summary=f"Evidence convergence on '{label}'",
                rc=None,
                evidence_by_target=evidence_by_target,
                cluster_membership=cluster_membership,
                kg_node_map=kg_node_map,
                node_edges=node_edges,
                products=products,
                companies=companies,
                technologies=technologies,
            )
            candidates.append(cand)
            covered_problems.add(eid)

        # 3. Candidates from cluster representatives (top signals)
        for cl in clusters:
            rep = cl.representative_id if hasattr(cl, "representative_id") else ""
            if rep in covered_problems:
                continue
            if hasattr(cl, "member_count") and cl.member_count < self._config.min_cluster_size_for_opportunity:
                continue
            label = rep
            cand = self._make_candidate(
                root_problem=rep,
                title=label,
                summary=f"Cluster representative '{label}'",
                rc=None,
                evidence_by_target=evidence_by_target,
                cluster_membership=cluster_membership,
                kg_node_map=kg_node_map,
                node_edges=node_edges,
                products=products,
                companies=companies,
                technologies=technologies,
            )
            candidates.append(cand)
            covered_problems.add(rep)

        # Filter by minimum confidence
        min_conf = self._config.min_confidence_threshold
        candidates = [c for c in candidates if c.get("reasoning_confidence", 0) >= min_conf or c.get("evidence_count", 0) >= self._config.min_evidence_for_opportunity]

        # Compute platform counts
        self._compute_platform_data(candidates, kg_edges, kg_node_map)

        return candidates

    def _make_candidate(
        self,
        root_problem: str,
        title: str,
        summary: str,
        rc: Any | None,
        evidence_by_target: dict,
        cluster_membership: dict,
        kg_node_map: dict,
        node_edges: dict,
        products: set,
        companies: set,
        technologies: set,
    ) -> dict:
        evidence_ids: list[str] = []
        reasoning_chains: list[str] = []
        cluster_ids: list[str] = []
        path_nodes: set[str] = set()
        kg_node_ids: list[str] = []

        if rc is not None:
            if hasattr(rc, "path"):
                path_nodes.update(rc.path)
            if hasattr(rc, "evidence_count"):
                pass

        # Evidence linked to this problem
        for target, ev_list in evidence_by_target.items():
            if target == root_problem or target in path_nodes:
                for ev in ev_list:
                    eid = ev.conclusion_node_id if hasattr(ev, "conclusion_node_id") else ""
                    if eid:
                        evidence_ids.append(eid)

        # Clusters containing this problem's nodes
        all_nodes = set(path_nodes)
        all_nodes.add(root_problem)
        for nid in all_nodes:
            for cl in cluster_membership.get(nid, []):
                cid = cl.cluster_id if hasattr(cl, "cluster_id") else ""
                if cid and cid not in cluster_ids:
                    cluster_ids.append(cid)

        # KG nodes connected to problem
        visited: set[str] = set()
        queue = [root_problem]
        while queue and len(kg_node_ids) < 50:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            if nid in kg_node_map:
                kg_node_ids.append(nid)
            for edge in node_edges.get(nid, []):
                src = edge.source_node_id if hasattr(edge, "source_node_id") else ""
                tgt = edge.target_node_id if hasattr(edge, "target_node_id") else ""
                other = tgt if src == nid else src
                if other not in visited:
                    queue.append(other)

        # Entity extraction from KG nodes
        affected_products: list[str] = []
        affected_companies: list[str] = []
        affected_technologies: list[str] = []
        for nid in kg_node_ids:
            node = kg_node_map.get(nid)
            if node is None:
                continue
            nt = node.node_type.value if hasattr(node, "node_type") and hasattr(node.node_type, "value") else ""
            label = node.label if hasattr(node, "label") else ""
            if nt == "product" and label and label not in affected_products:
                affected_products.append(label)
            elif nt == "company" and label and label not in affected_companies:
                affected_companies.append(label)
            elif nt == "technology" and label and label not in affected_technologies:
                affected_technologies.append(label)

        # Scoring signals from root cause
        pain_severity = 0.5
        reasoning_confidence = 0.5
        transitive_impact = 1
        if rc is not None:
            pain_severity = getattr(rc, "propagated_confidence", 0.5)
            reasoning_confidence = getattr(rc, "propagated_confidence", 0.5)
            transitive_impact = getattr(rc, "transitive_impact_count", 1)

        # Cluster density signal
        cluster_density = 0.5
        if cluster_ids:
            densities = []
            for cid in cluster_ids:
                for cl in cluster_membership.get(root_problem, []):
                    if hasattr(cl, "density"):
                        densities.append(cl.density)
            if densities:
                cluster_density = sum(densities) / len(densities)

        platform_nodes = [n for n in kg_node_ids if kg_node_map.get(n) and hasattr(kg_node_map[n], "node_type") and hasattr(kg_node_map[n].node_type, "value") and kg_node_map[n].node_type.value in ("source", "document")]
        platform_count = max(1, len(platform_nodes))

        return {
            "root_problem": root_problem,
            "title": title[:200],
            "summary": summary[:500],
            "evidence_ids": evidence_ids,
            "reasoning_chain_ids": reasoning_chains,
            "cluster_ids": cluster_ids,
            "kg_node_ids": kg_node_ids,
            "affected_products": affected_products,
            "affected_companies": affected_companies,
            "affected_technologies": affected_technologies,
            "pain_severity": pain_severity,
            "frequency_score": min(1.0, len(evidence_ids) / max(len(evidence_ids), 5)),
            "trend_score": 0.5,
            "evidence_count": len(evidence_ids),
            "reasoning_confidence": reasoning_confidence,
            "transitive_impact": transitive_impact,
            "product_count": len(affected_products),
            "company_count": len(affected_companies),
            "technology_count": len(affected_technologies),
            "platform_count": platform_count,
            "cluster_density": cluster_density,
            "competition_score": 0.5,
            "feasibility_score": 0.5,
            "novelty_score": 0.5,
            "recent_evidence_ratio": 0.5,
            "evidence_growth_rate": 0.0,
            "estimated_market_size": MarketSize.UNKNOWN.value,
        }

    def _compute_platform_data(
        self, candidates: list[dict], kg_edges: list, kg_node_map: dict
    ) -> None:
        for cand in candidates:
            platform_nodes = set()
            for nid in cand.get("kg_node_ids", []):
                for e in kg_edges:
                    src = e.source_node_id if hasattr(e, "source_node_id") else ""
                    tgt = e.target_node_id if hasattr(e, "target_node_id") else ""
                    if src == nid or tgt == nid:
                        other = tgt if src == nid else src
                        node = kg_node_map.get(other)
                        if node and hasattr(node, "source_asset"):
                            platform_nodes.add(node.source_asset)
            cand["platform_count"] = max(1, len(platform_nodes))
