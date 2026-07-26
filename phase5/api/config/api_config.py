from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from phase5.api.config.settings import APISettings


class UnifiedAPIConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    api: APISettings

    knowledge_dir: Path = Path("pain_intelligence/knowledge/assets/phase3")
    config_path: Path = Path("configs/default.yaml")

    enabled_capabilities: dict[str, bool] = Field(default_factory=lambda: {
        "knowledge_graph": True,
        "reasoning": True,
        "opportunity": True,
        "trend": True,
        "presentation": True,
        "copilot": True,
        "streaming": True,
    })

    enabled_exports: list[str] = Field(default_factory=lambda: ["json", "markdown", "pdf", "docx", "pptx"])


_runtime_config: dict[str, Any] = {}


def get_unified_config() -> dict[str, Any]:
    if _runtime_config:
        return _runtime_config
    return {"status": "loaded_from_defaults"}


def update_unified_config(updates: dict[str, Any]) -> dict[str, Any]:
    _runtime_config.update(updates)
    return _runtime_config


def load_unified_config(yaml_path: str | Path | None = None) -> UnifiedAPIConfig:
    if yaml_path is None:
        yaml_path = Path("configs/api.yaml")
    p = Path(yaml_path)
    raw: dict[str, Any] = {}
    if p.exists():
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        raw = raw.get("api_config", raw)
    api_settings = APISettings()
    if "api" in raw:
        api_settings = APISettings(**raw["api"])
    return UnifiedAPIConfig(api=api_settings, **{k: v for k, v in raw.items() if k != "api"})
