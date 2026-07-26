from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class HealthResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = "ok"
    version: str = "4.0.0"
    schema_version: str = "4.0"
    pipeline_version: str = "4.0"


class ReadinessResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ready: bool = True
    modules: dict[str, str] = {}
