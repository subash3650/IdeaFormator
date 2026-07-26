from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Permission(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    resource: str
    action: str


class TokenPayload(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sub: str
    roles: list[str] = Field(default_factory=list)
    permissions: list[str] = Field(default_factory=list)
    exp: int = 0
    iat: int = 0


class UserContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = "anonymous"
    roles: list[str] = Field(default_factory=lambda: ["anonymous"])
    permissions: list[str] = Field(default_factory=list)
    token_type: str = "none"
