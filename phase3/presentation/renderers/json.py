from __future__ import annotations

import json
from typing import Any

from phase3.presentation.config import PresentationConfig
from phase3.presentation.renderers.base import Renderer
from phase3.presentation.renderers.registry import register_renderer
from phase3.presentation.schema import PresentationModel, ReportFormat


@register_renderer(name="json")
class JSONRenderer(Renderer):
    @property
    def name(self) -> str:
        return "json"

    @property
    def format(self) -> ReportFormat:
        return ReportFormat.json

    def render(self, model: PresentationModel, config: PresentationConfig) -> str:
        data = model.model_dump(mode="json")
        return json.dumps(data, indent=2, default=str)
