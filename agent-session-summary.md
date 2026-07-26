## Objective
- Complete Phase 3 Module 5 — Presentation Layer under `phase3/presentation/` — the final output layer that aggregates KG, reasoning, opportunity, and trend data into multi-format intelligence reports with templated sections, chart providers, and renderers.

## Important Details
- All existing patterns followed: Pydantic v2 frozen/extra-forbid, decorator-based registries (renderers, templates, chart providers), Typer CLI with Rich tables, deterministic SHA-256 IDs, Parquet + JSON store.
- Presentation Layer is the **final output consumer** — reads from all upstream modules via `DataCollector`, never writes to them.
- Reports support 7 formats (json, markdown, html, csv, pdf, docx, pptx), 6 templates (executive, investor, founder, business, technology, market), 7 chart types (bar, line, pie, timeline, heatmap, treemap, sankey), and 11 section types.
- CLI registered as `pain presentation` under the main `pain-intelligence` app.

## Work State
### Completed
- **Schema** (`schema.py`): 9 enums, 19 Pydantic v2 models (PresentationModel, ReportSection, ReportAssets, ChartSpec, TableSpec, ReportOutput, ReportIndex, ReportComparison, etc.), auto-generated IDs, deterministic checksums.
- **Config** (`config.py`): PresentationConfig with YAML loading, all display thresholds, path resolution for phase2/phase3 dirs.
- **Store** (`store.py`): PresentationStore with Parquet report index, JSON content/metadata/manifest, SHA-256 checksums, content directory per report.
- **Collector** (`collector.py`): DataCollector with 4 methods (collect_kg, collect_reasoning, collect_opportunities, collect_trends), graceful fallback on missing upstream modules.
- **Builder** (`builder.py`): PresentationModelBuilder with 10 section builders, template-driven section ordering, asset/metrics/highlights extraction, summary generation (paragraph, bullets, sentence, tweet), full pipeline orchestration.
- **Providers** (`providers/`): 7 chart providers (bar, line, pie, timeline, heatmap, treemap, sankey) — each converts dict data to ChartSpec, priority-aware registry.
- **Renderers** (`renderers/`): 5 renderers (json, markdown, html, csv, plotly) — full Markdown/HTML report generation, Jinja2 HTML with fallback, Plotly figures for all 7 chart types, CSV findings export.
- **Templates** (`templates/`): 6 templates (business, executive, investor, founder, technology, market) — each defines sections/chart types/title/subtitle, registry pattern.
- **Engine** (`engine.py`): PresentationEngine facade — generate(), stats(), list_reports(), get_report(), search(), compare(), export(), clear_cache().
- **Search** (`search.py`): PresentationSearch — 9 search methods with text/type/tag/company/technology/date filtering.
- **Exporter** (`exporter.py`): PresentationExporter — export reports to filesystem in any format, export_all, export_to_format, export_summary.
- **CLI** (`cli.py`): 6 Typer commands (generate, list, show, export, stats, search) with Rich tables.
- **Main CLI registration**: `presentation_app` added to `pain_intelligence/cli.py` as `pain presentation ...`.
- **`__init__.py`**: Exposes all 19 Pydantic models + PresentationConfig + engine/search/exporter/store + `presentation_app`.
- **Tests**: 283 tests across 8 test files covering config (30), schema (38), store (30), renderers (33), templates (44), providers (32), plotly (16), templates ordering (3). All 283 pass. All 182 trend tests also pass. Zero regressions.

### Active
- (none)

### Blocked
- (none)

## Next Move
- (none — Phase 3 Module 5 implementation complete)

## Relevant Files
- `phase3/presentation/__init__.py` — module exports + `__version__ = "1.0.0"`
- `phase3/presentation/schema.py` — 19 Pydantic v2 models for the presentation layer
- `phase3/presentation/config.py` — PresentationConfig with YAML loading, all thresholds
- `phase3/presentation/store.py` — PresentationStore with Parquet + JSON storage
- `phase3/presentation/collector.py` — DataCollector (kg, reasoning, opportunities, trends)
- `phase3/presentation/builder.py` — PresentationModelBuilder (10 section builders)
- `phase3/presentation/engine.py` — PresentationEngine facade
- `phase3/presentation/search.py` — PresentationSearch (9 methods)
- `phase3/presentation/exporter.py` — PresentationExporter (4 export methods)
- `phase3/presentation/cli.py` — 6 Typer CLI commands
- `phase3/presentation/providers/` — 7 chart providers + ABC + priority registry
- `phase3/presentation/renderers/` — 5 renderers (json, markdown, html, csv, plotly)
- `phase3/presentation/templates/` — 6 templates (business, executive, investor, founder, technology, market)
- `phase3/presentation/tests/` — 283 tests across 8 files
- `pain_intelligence/cli.py` — `presentation_app` registered at line 35
