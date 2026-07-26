from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from phase4.copilot.schema import Intent, ToolMetadata, ToolResult, PermissionType, ToolPriority


class BaseTool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def metadata(self) -> ToolMetadata:
        ...

    @abstractmethod
    def execute(self, params: dict[str, Any]) -> ToolResult:
        ...

    def can_handle(self, intent: Intent) -> bool:
        return intent in self.metadata.supported_intents
