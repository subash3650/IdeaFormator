from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from phase3.presentation.config import PresentationConfig, load_presentation_config
from phase3.presentation.schema import ReportFormat, ReportType


class TestPresentationConfig:
    def test_default_config(self) -> None:
        cfg = PresentationConfig()
        assert cfg.output_dir == Path("pain_intelligence/knowledge/assets/phase3")
        assert cfg.reports_dir == Path("reports")
        assert cfg.default_template == "executive"
        assert cfg.max_findings == 20
        assert cfg.max_charts == 15
        assert cfg.chart_width == 800
        assert cfg.chart_height == 500
        assert cfg.chart_theme == "plotly_white"
        assert cfg.evaluation_enabled is True
        assert cfg.enable_scheduler is False

    def test_default_formats(self) -> None:
        cfg = PresentationConfig()
        assert ReportFormat.json in cfg.enabled_formats
        assert ReportFormat.markdown in cfg.enabled_formats
        assert ReportFormat.html in cfg.enabled_formats
        assert ReportFormat.csv in cfg.enabled_formats
        assert ReportFormat.pdf not in cfg.enabled_formats

    def test_default_report_types(self) -> None:
        cfg = PresentationConfig()
        assert len(cfg.enabled_report_types) == len(ReportType)

    def test_frozen(self) -> None:
        cfg = PresentationConfig()
        with pytest.raises(ValidationError):
            cfg.max_findings = 50

    def test_extra_forbid(self) -> None:
        with pytest.raises(ValidationError):
            PresentationConfig(unknown_field="x")

    def test_custom_values(self) -> None:
        cfg = PresentationConfig(
            max_findings=50,
            max_charts=10,
            default_template="investor",
            chart_theme="plotly_dark",
            enable_scheduler=True,
        )
        assert cfg.max_findings == 50
        assert cfg.max_charts == 10
        assert cfg.default_template == "investor"
        assert cfg.chart_theme == "plotly_dark"
        assert cfg.enable_scheduler is True

    def test_formats_override(self) -> None:
        cfg = PresentationConfig(
            enabled_formats=[ReportFormat.json, ReportFormat.pdf]
        )
        assert cfg.enabled_formats == [ReportFormat.json, ReportFormat.pdf]

    def test_report_types_override(self) -> None:
        cfg = PresentationConfig(
            enabled_report_types=[ReportType.executive_summary, ReportType.weekly]
        )
        assert len(cfg.enabled_report_types) == 2

    def test_path_coercion(self) -> None:
        cfg = PresentationConfig(output_dir="custom/path")
        assert isinstance(cfg.output_dir, Path)
        assert cfg.output_dir == Path("custom/path")

    @pytest.mark.parametrize("field", ["max_findings", "max_charts", "max_trends_displayed"])
    def test_bounds_low(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: -1}
        with pytest.raises(ValidationError):
            PresentationConfig(**kwargs)

    @pytest.mark.parametrize("field", ["chart_width", "chart_height"])
    def test_bounds_high(self, field: str) -> None:
        kwargs: dict[str, Any] = {field: 99999}
        with pytest.raises(ValidationError):
            PresentationConfig(**kwargs)


class TestProperties:
    def test_report_dir(self) -> None:
        cfg = PresentationConfig()
        assert cfg.report_dir == Path("pain_intelligence/knowledge/assets/phase3/reports")

    def test_report_dir_with_knowledge_dir(self) -> None:
        cfg = PresentationConfig(knowledge_dir=Path("custom/knowledge"))
        assert cfg.report_dir == Path("custom/knowledge/reports")

    def test_phase2_dir(self) -> None:
        cfg = PresentationConfig()
        assert cfg.phase2_dir == Path("pain_intelligence/knowledge/assets/phase2")

    def test_phase3_dir(self) -> None:
        cfg = PresentationConfig()
        assert cfg.phase3_dir == Path("pain_intelligence/knowledge/assets/phase3")


class TestLoadConfig:
    def test_load_nonexistent_file_returns_default(self) -> None:
        cfg = load_presentation_config("nonexistent_config.yaml")
        assert isinstance(cfg, PresentationConfig)
        assert cfg.max_findings == 20

    def test_load_none_path_returns_default(self) -> None:
        cfg = load_presentation_config(None)
        assert isinstance(cfg, PresentationConfig)

    def test_load_from_file(self, tmp_path: Path) -> None:
        config_data = {
            "presentation": {
                "max_findings": 30,
                "default_template": "investor",
                "chart_theme": "plotly_dark",
                "evaluation_enabled": False,
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(str(config_path), "w") as f:
            yaml.dump(config_data, f)

        cfg = load_presentation_config(str(config_path))
        assert cfg.max_findings == 30
        assert cfg.default_template == "investor"
        assert cfg.chart_theme == "plotly_dark"
        assert cfg.evaluation_enabled is False

    def test_load_with_enums(self, tmp_path: Path) -> None:
        config_data = {
            "presentation": {
                "enabled_formats": ["json", "html"],
                "default_format": "markdown",
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(str(config_path), "w") as f:
            yaml.dump(config_data, f)

        cfg = load_presentation_config(str(config_path))
        assert cfg.enabled_formats == [ReportFormat.json, ReportFormat.html]
        assert cfg.default_format == ReportFormat.markdown

    def test_load_with_report_types(self, tmp_path: Path) -> None:
        config_data = {
            "presentation": {
                "enabled_report_types": ["weekly", "monthly"],
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(str(config_path), "w") as f:
            yaml.dump(config_data, f)

        cfg = load_presentation_config(str(config_path))
        assert cfg.enabled_report_types == [ReportType.weekly, ReportType.monthly]

    def test_load_with_paths(self, tmp_path: Path) -> None:
        config_data = {
            "presentation": {
                "output_dir": "custom/output",
                "knowledge_dir": "custom/knowledge",
            }
        }
        config_path = tmp_path / "config.yaml"
        with open(str(config_path), "w") as f:
            yaml.dump(config_data, f)

        cfg = load_presentation_config(str(config_path))
        assert cfg.output_dir == Path("custom/output")
        assert cfg.knowledge_dir == Path("custom/knowledge")

    def test_empty_presentation_section(self, tmp_path: Path) -> None:
        config_data = {"other_section": {"key": "val"}}
        config_path = tmp_path / "config.yaml"
        with open(str(config_path), "w") as f:
            yaml.dump(config_data, f)

        cfg = load_presentation_config(str(config_path))
        assert cfg.max_findings == 20

    def test_empty_yaml(self, tmp_path: Path) -> None:
        config_path = tmp_path / "empty.yaml"
        with open(str(config_path), "w") as f:
            f.write("")
        cfg = load_presentation_config(str(config_path))
        assert isinstance(cfg, PresentationConfig)


class TestEdgeCases:
    def test_zero_charts(self) -> None:
        cfg = PresentationConfig(max_charts=3)
        assert cfg.max_charts == 3

    def test_max_root_causes(self) -> None:
        cfg = PresentationConfig(max_root_causes_displayed=30)
        assert cfg.max_root_causes_displayed == 30
        with pytest.raises(ValidationError):
            PresentationConfig(max_root_causes_displayed=100)

    def test_schedules_default(self) -> None:
        cfg = PresentationConfig()
        assert cfg.schedules == []

    def test_html_customization(self) -> None:
        cfg = PresentationConfig(html_title="Custom", html_author="Author")
        assert cfg.html_title == "Custom"
        assert cfg.html_author == "Author"

    def test_default_template(self) -> None:
        cfg = PresentationConfig()
        assert cfg.default_template in cfg.enabled_templates

    def test_template_list(self) -> None:
        cfg = PresentationConfig(enabled_templates=["executive"])
        assert cfg.enabled_templates == ["executive"]
