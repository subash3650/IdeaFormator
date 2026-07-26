"""OpportunitySearch — query and filter discovered opportunities."""

from __future__ import annotations

from phase3.opportunity.schema import Opportunity, OpportunityType, RecommendationType


class OpportunitySearch:
    """Search API for filtering discovered opportunities."""

    def __init__(self, opportunities: list[Opportunity]) -> None:
        self._opportunities = opportunities

    def find_by_id(self, opportunity_id: str) -> Opportunity | None:
        for o in self._opportunities:
            if o.opportunity_id == opportunity_id:
                return o
        return None

    def find_by_entity(self, entity_name: str) -> list[Opportunity]:
        query = entity_name.lower()
        result: list[Opportunity] = []
        for o in self._opportunities:
            for lst in [o.affected_products, o.affected_companies, o.affected_technologies]:
                if any(query in item.lower() for item in lst):
                    result.append(o)
                    break
        return result

    def find_by_company(self, company: str) -> list[Opportunity]:
        query = company.lower()
        return [
            o for o in self._opportunities
            if any(query in c.lower() for c in o.affected_companies)
        ]

    def find_by_product(self, product: str) -> list[Opportunity]:
        query = product.lower()
        return [
            o for o in self._opportunities
            if any(query in p.lower() for p in o.affected_products)
        ]

    def find_by_cluster(self, cluster_id: str) -> list[Opportunity]:
        return [o for o in self._opportunities if cluster_id in o.cluster_ids]

    def find_by_reasoning_chain(self, chain_id: str) -> list[Opportunity]:
        return [o for o in self._opportunities if chain_id in o.reasoning_chain_ids]

    def find_by_platform(self, platform: str) -> list[Opportunity]:
        query = platform.lower()
        return [
            o for o in self._opportunities
            if any(query in nid.lower() for nid in o.kg_node_ids)
        ]

    def find_cross_platform(self) -> list[Opportunity]:
        return [o for o in self._opportunities if o.scoring_breakdown.cross_platform >= 0.5]

    def find_emerging(self, min_trend: float = 0.6) -> list[Opportunity]:
        return [
            o for o in self._opportunities
            if o.trend_score >= min_trend and o.recommendation_type
            in (RecommendationType.STRONG_PURSUE, RecommendationType.WORTH_EXPLORING)
        ]

    def search_text(self, query: str, top_k: int = 10) -> list[Opportunity]:
        if not query.strip():
            return []
        query_lower = query.lower()
        scored: list[tuple[Opportunity, float]] = []
        for o in self._opportunities:
            score = 0.0
            if query_lower in o.title.lower():
                score += 3.0
            if query_lower in o.summary.lower():
                score += 2.0
            if query_lower in o.root_problem.lower():
                score += 2.0
            for lst in [o.affected_products, o.affected_companies, o.affected_technologies]:
                if any(query_lower in item.lower() for item in lst):
                    score += 1.0
                    break
            if score > 0:
                scored.append((o, score))
        scored.sort(key=lambda x: -x[1])
        return [o for o, _ in scored[:top_k]]
