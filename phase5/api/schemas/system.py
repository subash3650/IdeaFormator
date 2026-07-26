from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SystemInfoResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "4.0.0"
    schema_version: str = "4.0"
    pipeline_version: str = "4.0"
    environment: str = "development"
    python_version: str = ""
    modules: list[str] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: dict[str, bool] = Field(default_factory=dict)
    exports: list[str] = Field(default_factory=list)


class VersionResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: str = "4.0.0"
    schema_version: str = "4.0"
    pipeline_version: str = "4.0"
    build: str = ""
