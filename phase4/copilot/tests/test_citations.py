from __future__ import annotations

from phase4.copilot.citations.citation_system import CitationBuilder
from phase4.copilot.schema import Citation, CitationSource, ToolResult


class TestCitationBuilder:
    def setup_method(self):
        self.builder = CitationBuilder(min_confidence=0.0)

    def _make_citation(self, cid: str = "c1", conf: float = 0.8) -> Citation:
        return Citation(
            source_module=CitationSource.KNOWLEDGE_GRAPH,
            source_id=f"node_{cid}",
            source_title=f"Node {cid}",
            confidence=conf,
            snippet="Test snippet",
        )

    def _make_tool_result(self, citations: list[Citation]) -> ToolResult:
        return ToolResult(tool_name="test", data={}, citations=citations)

    def test_from_tool_result(self):
        c = self._make_citation()
        tr = self._make_tool_result([c])
        result = self.builder.from_tool_result(tr)
        assert len(result) == 1

    def test_from_multiple_results(self):
        c1 = self._make_citation("c1")
        c2 = self._make_citation("c2")
        r1 = self._make_tool_result([c1])
        r2 = self._make_tool_result([c2])
        results = self.builder.from_results([r1, r2])
        assert len(results) == 2

    def test_deduplication(self):
        c = self._make_citation("c1")
        r1 = self._make_tool_result([c])
        r2 = self._make_tool_result([c])
        results = self.builder.from_results([r1, r2])
        assert len(results) == 1

    def test_filter_citations_low_confidence(self):
        builder = CitationBuilder(min_confidence=0.5)
        c1 = self._make_citation("c1", 0.1)
        c2 = self._make_citation("c2", 0.9)
        filtered = builder.filter_citations([c1, c2])
        assert len(filtered) == 1
        assert filtered[0].citation_id == c2.citation_id

    def test_filter_max_citations(self):
        builder = CitationBuilder(min_confidence=0.0, max_citations=2)
        citations = [self._make_citation(f"c{i}", 0.5) for i in range(10)]
        filtered = builder.filter_citations(citations)
        assert len(filtered) == 2

    def test_format_markdown(self):
        c = self._make_citation("c1", 0.85)
        md = self.builder.format_markdown([c])
        assert "Sources" in md
        assert c.source_title in md

    def test_format_markdown_empty(self):
        md = self.builder.format_markdown([])
        assert md == ""

    def test_format_json(self):
        c = self._make_citation("c1", 0.75)
        result = self.builder.format_json([c])
        assert len(result) == 1
        assert result[0]["source"] == "knowledge_graph"

    def test_unique_by_source_key(self):
        c = Citation(
            source_module=CitationSource.KNOWLEDGE_GRAPH,
            source_id="same_id",
            source_title="Same",
            confidence=0.5,
        )
        filtered = self.builder.filter_citations([c, c])
        assert len(filtered) == 1
