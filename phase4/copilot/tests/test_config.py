from __future__ import annotations

from pathlib import Path

from phase4.copilot.config import CopilotConfig, load_copilot_config
from phase4.copilot.schema import ResponseFormat


class TestCopilotConfig:
    def test_default_config(self):
        config = CopilotConfig()
        assert config.planner_max_steps == 5
        assert config.planner_confidence_threshold == 0.4
        assert config.llm_provider == "mock"
        assert config.default_response_format == ResponseFormat.MARKDOWN

    def test_default_tools(self):
        config = CopilotConfig()
        assert "knowledge_graph" in config.enabled_tools
        assert "opportunity" in config.enabled_tools
        assert len(config.enabled_tools) == 8

    def test_default_memory_limits(self):
        config = CopilotConfig()
        assert config.max_short_term == 20
        assert config.max_conversation == 50
        assert config.max_long_term == 100
        assert config.compression_threshold == 100

    def test_default_citation_settings(self):
        config = CopilotConfig()
        assert config.min_citation_confidence == 0.2
        assert config.max_citations_per_response == 10

    def test_custom_values(self):
        config = CopilotConfig(
            planner_max_steps=3,
            llm_provider="openai",
            llm_model="gpt-4",
            planner_confidence_threshold=0.5,
        )
        assert config.planner_max_steps == 3
        assert config.planner_confidence_threshold == 0.5

    def test_frozen(self):
        config = CopilotConfig()
        try:
            config.llm_provider = "openai"
            assert False, "Should be frozen"
        except Exception:
            pass

    def test_output_dir_default(self):
        from pathlib import Path
        config = CopilotConfig()
        assert Path("pain_intelligence/knowledge/assets/phase3") == config.output_dir

    def test_phase2_dir_property(self):
        config = CopilotConfig(output_dir=Path("test/output"))
        assert "phase2" in str(config.phase2_dir)

    def test_phase3_dir_property(self):
        config = CopilotConfig(output_dir=Path("test/output"))
        assert str(config.phase3_dir) == "test\\output"

    def test_copilot_dir_property(self):
        config = CopilotConfig(output_dir=Path("test/output"))
        assert "copilot" in str(config.copilot_dir)

    def test_bounds_low(self):
        try:
            CopilotConfig(planner_max_steps=0)
            assert False
        except Exception:
            pass

    def test_bounds_high(self):
        try:
            CopilotConfig(planner_max_steps=20)
            assert False
        except Exception:
            pass

    def test_enabled_formats_default(self):
        config = CopilotConfig()
        assert config.enable_streaming is True
        assert config.enable_suggested_followups is True


class TestLoadConfig:
    def test_load_nonexistent_file_returns_default(self):
        config = load_copilot_config("nonexistent_file.yaml")
        assert config.planner_max_steps == 5

    def test_load_none_path_returns_default(self):
        config = load_copilot_config(None)
        assert config.llm_provider == "mock"
