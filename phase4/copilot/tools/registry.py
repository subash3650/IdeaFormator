from __future__ import annotations

from typing import Any

from phase4.copilot.schema import Intent, ToolMetadata
from phase4.copilot.tools.base import BaseTool


_registry: dict[str, type[BaseTool]] = {}
_metadata: dict[str, ToolMetadata] = {}


def register_tool(name: str, metadata: ToolMetadata | None = None):
    def decorator(cls: type[BaseTool]) -> type[BaseTool]:
        _registry[name] = cls
        return cls
    return decorator


def set_tool_metadata(name: str, metadata: ToolMetadata) -> None:
    _metadata[name] = metadata


def get_tool_metadata(name: str) -> ToolMetadata | None:
    return _metadata.get(name)


def create_tool(name: str, **kwargs: Any) -> BaseTool:
    cls = _registry.get(name)
    if cls is None:
        msg = f"Unknown tool '{name}'. Available: {', '.join(sorted(_registry))}"
        raise KeyError(msg)
    return cls(**kwargs)


def available_tools() -> list[str]:
    return sorted(_registry)


def tools_for_intent(intent: Intent) -> list[str]:
    return sorted(
        name for name in _registry
        if intent in _metadata.get(name, ToolMetadata(name="", description="", supported_intents=[])).supported_intents
    )


def tools_with_capability(capability: str) -> list[str]:
    result: list[str] = []
    for name, meta in _metadata.items():
        if capability in meta.description.lower():
            result.append(name)
    return sorted(result)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, type[BaseTool]] = {}
        self._metadata: dict[str, ToolMetadata] = {}

    def register(self, name: str, cls: type[BaseTool], metadata: ToolMetadata | None = None) -> None:
        self._tools[name] = cls
        if metadata is not None:
            self._metadata[name] = metadata

    def create(self, name: str, **kwargs: Any) -> BaseTool:
        cls = self._tools.get(name)
        if cls is None:
            msg = f"Unknown tool '{name}'"
            raise KeyError(msg)
        return cls(**kwargs)

    def list_tools(self) -> list[str]:
        return sorted(self._tools)

    def get_metadata(self, name: str) -> ToolMetadata | None:
        return self._metadata.get(name)


import phase4.copilot.tools.kg_tool  # noqa: E402, F401
import phase4.copilot.tools.reasoning_tool  # noqa: E402, F401
import phase4.copilot.tools.opportunity_tool  # noqa: E402, F401
import phase4.copilot.tools.trend_tool  # noqa: E402, F401
import phase4.copilot.tools.presentation_tool  # noqa: E402, F401
import phase4.copilot.tools.search_tool  # noqa: E402, F401
import phase4.copilot.tools.comparison_tool  # noqa: E402, F401
import phase4.copilot.tools.evidence_tool  # noqa: E402, F401
