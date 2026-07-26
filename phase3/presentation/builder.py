from __future__ import annotations

import time
from typing import Any

from phase3.presentation.collector import DataCollector
from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers import available_renderers, create_renderer
from phase3.presentation.schema import (
    ChartSpec,
    ChartType,
    Highlight,
    PresentationModel,
    ReportAssets,
    ReportFormat,
    ReportIndexEntry,
    ReportOutput,
    ReportSection,
    ReportSummaries,
    ReportVersion,
    SectionDefinition,
    SectionProvenance,
    SectionType,
    SourceLineage,
    TableSpec,
)
from phase3.presentation.store import PresentationStore
from phase3.presentation.templates import available_templates, create_template


class PresentationModelBuilder:
    def __init__(self, config: PresentationConfig) -> None:
        self._config = config
        self._store = PresentationStore(config.output_dir)
        self._collector = DataCollector(config)

    @property
    def store(self) -> PresentationStore:
        return self._store

    def build(
        self,
        report_type: str = "executive_summary",
        template_name: str | None = None,
        output_formats: list[ReportFormat] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        start = time.perf_counter()

        if template_name is None:
            template_name = self._config.default_template

        if output_formats is None:
            output_formats = self._config.enabled_formats

        collected = self._collector.collect_all()
        template = create_template(template_name)
        template.configure(self._config)

        sections = self._build_sections(template, collected)
        assets = self._build_assets(collected, template)
        summaries = self._build_summaries(collected, sections)
        lineage = self._build_lineage(collected)

        report_title = template.title(collected)
        report_subtitle = template.subtitle(collected)

        model = PresentationModel(
            report_type=report_type,
            title=report_title,
            subtitle=report_subtitle,
            versions=ReportVersion(
                template_version=self._get_template_version(template),
            ),
            lineage=lineage,
            sections=sections,
            assets=assets,
            summaries=summaries,
            tags=self._extract_tags(collected),
            companies=self._extract_companies(collected),
            technologies=self._extract_technologies(collected),
            products=self._extract_products(collected),
        )

        rendered_formats: list[ReportFormat] = []
        checksums_dict: dict[str, str] = {}
        for fmt in output_formats:
            renderer_name = fmt.value
            if renderer_name in available_renderers():
                try:
                    renderer = create_renderer(renderer_name)
                    output = renderer.render(model, self._config)
                    rendered_formats.append(fmt)
                except Exception:
                    pass

        self._store.save_content(model)
        self._store.save_report(self._build_output(model, rendered_formats, time.perf_counter() - start))
        self._store.save_metadata(self._build_metadata(collected))

        elapsed = time.perf_counter() - start

        return {
            "report_id": model.report_id,
            "report_type": report_type,
            "title": report_title,
            "sections_count": len(sections),
            "charts_count": len(assets.charts),
            "formats": [f.value for f in rendered_formats],
            "elapsed_seconds": round(elapsed, 3),
        }

    def _build_sections(self, template: Any, collected: dict[str, Any]) -> list[ReportSection]:
        sections: list[ReportSection] = []
        template_sections: list[SectionDefinition] = template.sections()

        section_builders = {
            SectionType.executive_summary: self._build_executive_summary,
            SectionType.top_findings: self._build_top_findings,
            SectionType.trend_analysis: self._build_trend_analysis,
            SectionType.opportunity_analysis: self._build_opportunity_analysis,
            SectionType.reasoning_summary: self._build_reasoning_summary,
            SectionType.root_causes: self._build_root_causes,
            SectionType.evidence: self._build_evidence,
            SectionType.confidence: self._build_confidence,
            SectionType.charts: self._build_charts,
            SectionType.recommendations: self._build_recommendations,
            SectionType.appendix: self._build_appendix,
        }

        for sec_def in template_sections:
            builder = section_builders.get(sec_def.section_type)
            if builder:
                section = builder(sec_def, collected)
                sections.append(section)

        return sorted(sections, key=lambda s: s.order)

    def _build_executive_summary(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        opp = collected.get("opportunities", {})
        trend = collected.get("trends", {})
        kg = collected.get("kg", {})
        reasoning = collected.get("reasoning", {})

        content = {
            "summary": f"Analysis of {opp.get('total', 0)} opportunities, "
                       f"{trend.get('total', 0)} trends, "
                       f"{reasoning.get('root_cause_count', 0)} root causes, "
                       f"and {kg.get('node_count', 0)} knowledge graph nodes.",
            "metrics": {
                "opportunities": opp.get("total", 0),
                "trends": trend.get("total", 0),
                "growing_trends": trend.get("growing_count", 0),
                "strong_pursue_opps": opp.get("strong_pursue_count", 0),
                "root_causes": reasoning.get("root_cause_count", 0),
                "kg_nodes": kg.get("node_count", 0),
            },
        }

        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content=content,
        )

    def _build_top_findings(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        findings: list[dict[str, Any]] = []

        for opp in collected.get("opportunities", {}).get("top_opportunities", []):
            score = getattr(opp, "opportunity_score", 0)
            title = getattr(opp, "title", "Untitled")
            findings.append({"title": title, "score": score, "source": "opportunity"})

        for trend in collected.get("trends", {}).get("top_trends", []):
            score = getattr(getattr(trend, "metrics", None), "trend_score", 0)
            title = getattr(trend, "title", "Untitled")
            findings.append({"title": title, "score": score, "source": "trend"})

        findings.sort(key=lambda f: f["score"], reverse=True)
        findings = findings[: self._config.max_findings]

        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content={"findings": findings, "total": len(findings)},
        )

    def _build_trend_analysis(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        trend = collected.get("trends", {})
        content: dict[str, Any] = {
            "total": trend.get("total", 0),
            "distribution": {
                "growing": trend.get("growing_count", 0),
                "declining": trend.get("declining_count", 0),
                "emerging": trend.get("emerging_count", 0),
            },
        }

        trends_list = trend.get("top_trends", [])
        items = []
        for t in trends_list[: self._config.max_trends_displayed]:
            score = getattr(getattr(t, "metrics", None), "trend_score", 0)
            items.append({"title": getattr(t, "title", ""), "score": score})
        content["items"] = items

        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content=content,
        )

    def _build_opportunity_analysis(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        opp = collected.get("opportunities", {})
        content: dict[str, Any] = {
            "total": opp.get("total", 0),
            "strong_pursue": opp.get("strong_pursue_count", 0),
            "worth_exploring": opp.get("worth_exploring_count", 0),
            "avg_score": opp.get("avg_score", 0.0),
            "max_score": opp.get("max_score", 0.0),
        }

        items = []
        for o in opp.get("top_opportunities", [])[: self._config.max_opportunities_displayed]:
            items.append({
                "title": getattr(o, "title", ""),
                "score": getattr(o, "opportunity_score", 0),
                "recommendation": getattr(getattr(o, "recommendation_type", None), "value", ""),
            })
        content["items"] = items

        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content=content,
        )

    def _build_reasoning_summary(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        r = collected.get("reasoning", {})
        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content={
                "inferences": r.get("inference_count", 0),
                "chains": r.get("chain_count", 0),
                "root_causes": r.get("root_cause_count", 0),
                "evidence": r.get("evidence_count", 0),
            },
        )

    def _build_root_causes(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        reasoning = collected.get("reasoning", {})
        causes = reasoning.get("root_causes", [])
        items = []
        for c in causes[: self._config.max_root_causes_displayed]:
            label = getattr(c, "cause_label", getattr(c, "cause_node_id", "Unknown"))
            score = getattr(c, "ranking_score", getattr(c, "propagated_confidence", 0))
            items.append({"title": label, "score": score})

        provenance = SectionProvenance(
            reasoning_chain_ids=[getattr(c, "path", [])[0] if getattr(c, "path", None) else "" for c in causes[:5]],
            confidence=reasoning.get("stats", {}).get("avg_inference_confidence", 0),
        )

        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content={"items": items, "total": len(items)},
            provenance=provenance,
        )

    def _build_evidence(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        reasoning = collected.get("reasoning", {})
        evidence_list = reasoning.get("evidence_aggregations", [])
        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content={
                "total": len(evidence_list),
                "items": [
                    {
                        "title": getattr(e, "conclusion_label", ""),
                        "confidence": getattr(e, "aggregated_confidence", 0),
                        "evidence_count": getattr(e, "evidence_count", 0),
                    }
                    for e in evidence_list[: self._config.max_evidence_displayed]
                ],
            },
        )

    def _build_confidence(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        scores: dict[str, float] = {}
        if collected.get("reasoning", {}).get("stats", {}).get("avg_inference_confidence"):
            scores["reasoning"] = collected["reasoning"]["stats"]["avg_inference_confidence"]
        if collected.get("opportunities", {}).get("avg_score"):
            scores["opportunities"] = collected["opportunities"]["avg_score"]
        if collected.get("trends", {}).get("avg_score"):
            scores["trends"] = collected["trends"]["avg_score"]

        overall = sum(scores.values()) / len(scores) if scores else 0.0
        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content={"metrics": scores, "overall": round(overall, 3)},
        )

    def _build_charts(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content={"total": 0},
        )

    def _build_recommendations(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        recs: list[str] = []

        opp = collected.get("opportunities", {})
        if opp.get("strong_pursue_count", 0) > 0:
            recs.append(f"Pursue {opp['strong_pursue_count']} high-value opportunities identified")
        if opp.get("worth_exploring_count", 0) > 0:
            recs.append(f"Explore {opp['worth_exploring_count']} promising opportunities further")

        trend = collected.get("trends", {})
        if trend.get("growing_count", 0) > 0:
            recs.append(f"Monitor {trend['growing_count']} growing trends for strategic alignment")
        if trend.get("emerging_count", 0) > 0:
            recs.append(f"Watch {trend['emerging_count']} emerging trends for early signals")

        if not recs:
            recs.append("Continue monitoring pipeline for new opportunities and trends")

        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content={"recommendations": recs},
        )

    def _build_appendix(self, sec: SectionDefinition, collected: dict[str, Any]) -> ReportSection:
        return ReportSection(
            section_type=sec.section_type,
            title=sec.title,
            order=sec.order,
            content={
                "sources": {
                    "knowledge_graph": collected.get("kg", {}).get("available", False),
                    "reasoning": collected.get("reasoning", {}).get("available", False),
                    "opportunities": collected.get("opportunities", {}).get("available", False),
                    "trends": collected.get("trends", {}).get("available", False),
                }
            },
        )

    def _build_assets(self, collected: dict[str, Any], template: Any) -> ReportAssets:
        opp = collected.get("opportunities", {})
        trend = collected.get("trends", {})
        reasoning = collected.get("reasoning", {})

        metrics: dict[str, float] = {
            "opportunities": float(opp.get("total", 0)),
            "strong_pursue": float(opp.get("strong_pursue_count", 0)),
            "avg_opp_score": opp.get("avg_score", 0.0),
            "trends": float(trend.get("total", 0)),
            "growing_trends": float(trend.get("growing_count", 0)),
            "avg_trend_score": trend.get("avg_score", 0.0),
            "root_causes": float(reasoning.get("root_cause_count", 0)),
            "inferences": float(reasoning.get("inference_count", 0)),
        }

        highlights: list[Highlight] = []
        for o in opp.get("top_opportunities", [])[:3]:
            title = getattr(o, "title", "Opportunity")
            score = getattr(o, "opportunity_score", 0)
            highlights.append(Highlight(text=title, source="opportunity", score=score, section=SectionType.opportunity_analysis))

        tables: list[TableSpec] = []
        if trend.get("distribution"):
            dist = trend["distribution"]
            tables.append(TableSpec(
                title="Trend Distribution",
                headers=["Type", "Count"],
                rows=[[k, str(v)] for k, v in dist.items()],
            ))

        return ReportAssets(
            metrics=metrics,
            highlights=highlights,
            tables=tables,
        )

    def _build_summaries(self, collected: dict[str, Any], sections: list[ReportSection]) -> ReportSummaries:
        opp = collected.get("opportunities", {})
        trend = collected.get("trends", {})

        paragraph = (
            f"This report analyzes {opp.get('total', 0)} opportunities "
            f"and {trend.get('total', 0)} trends. "
            f"Found {opp.get('strong_pursue_count', 0)} strong-pursue opportunities "
            f"and {trend.get('growing_count', 0)} growing trends."
        )

        bullets = []
        if opp.get("strong_pursue_count", 0) > 0:
            bullets.append(f"{opp['strong_pursue_count']} opportunities recommended for immediate pursuit")
        if trend.get("growing_count", 0) > 0:
            bullets.append(f"{trend['growing_count']} growing trends detected")
        if trend.get("emerging_count", 0) > 0:
            bullets.append(f"{trend['emerging_count']} emerging trends identified")

        return ReportSummaries(
            one_paragraph=paragraph,
            five_bullets=bullets,
            one_sentence=f"Analysis of {opp.get('total', 0)} opportunities and {trend.get('total', 0)} trends.",
            one_tweet=f"IdeaFormator report: {opp.get('total', 0)} opportunities, {trend.get('total', 0)} trends analyzed.",
        )

    def _build_lineage(self, collected: dict[str, Any]) -> SourceLineage:
        return SourceLineage(
            knowledge_graph_run_id=None,
            reasoning_run_id=None,
            opportunity_run_id=None,
            trend_run_id=None,
        )

    def _build_output(self, model: PresentationModel, formats: list[ReportFormat], elapsed: float) -> ReportOutput:
        entry = ReportIndexEntry(
            report_id=model.report_id,
            report_type=model.report_type,
            title=model.title,
            generated_at=model.generated_at,
            tags=model.tags,
            companies=model.companies,
            technologies=model.technologies,
            products=model.products,
            formats=formats,
            sections=[s.section_type for s in model.sections],
        )

        return ReportOutput(
            report_id=model.report_id,
            report_type=model.report_type,
            title=model.title,
            generated_at=model.generated_at,
            sections_count=len(model.sections),
            charts_count=len(model.assets.charts),
            formats=formats,
            index_entry=entry,
            elapsed_seconds=elapsed,
        )

    def _build_metadata(self, collected: dict[str, Any]) -> dict[str, Any]:
        return {
            "generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "sources": {
                "knowledge_graph": collected.get("kg", {}).get("available", False),
                "reasoning": collected.get("reasoning", {}).get("available", False),
                "opportunities": collected.get("opportunities", {}).get("available", False),
                "trends": collected.get("trends", {}).get("available", False),
            },
        }

    def _get_template_version(self, template: Any) -> str:
        return getattr(template, "_config", None) and getattr(template._config, "version", "1.0.0") or "1.0.0"

    def _extract_tags(self, collected: dict[str, Any]) -> list[str]:
        tags: list[str] = []
        if collected.get("trends", {}).get("available"):
            tags.append("trends")
        if collected.get("opportunities", {}).get("available"):
            tags.append("opportunities")
        if collected.get("reasoning", {}).get("available"):
            tags.append("reasoning")
        if collected.get("kg", {}).get("available"):
            tags.append("knowledge_graph")
        return tags

    def _extract_companies(self, collected: dict[str, Any]) -> list[str]:
        companies: set[str] = set()
        for opp in collected.get("opportunities", {}).get("opportunities", []):
            affected = getattr(opp, "affected_companies", [])
            if isinstance(affected, list):
                companies.update(affected)
        return sorted(companies)

    def _extract_technologies(self, collected: dict[str, Any]) -> list[str]:
        technologies: set[str] = set()
        for opp in collected.get("opportunities", {}).get("opportunities", []):
            affected = getattr(opp, "affected_technologies", [])
            if isinstance(affected, list):
                technologies.update(affected)
        return sorted(technologies)

    def _extract_products(self, collected: dict[str, Any]) -> list[str]:
        products: set[str] = set()
        for opp in collected.get("opportunities", {}).get("opportunities", []):
            affected = getattr(opp, "affected_products", [])
            if isinstance(affected, list):
                products.update(affected)
        return sorted(products)
