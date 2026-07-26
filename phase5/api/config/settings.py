from __future__ import annotations

from pathlib import Path
from typing import Any
import os

import yaml
from pydantic import BaseModel, ConfigDict, Field


class CacheSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    l1_ttl_seconds: int = 60
    l2_ttl_seconds: int = 300
    l1_max_size: int = 5000
    redis_url: str = "redis://localhost:6379/0"
    redis_enabled: bool = True


class RateLimitSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    redis_enabled: bool = True


class CORSSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_origins: list[str] = Field(default_factory=lambda: ["*"])
    allow_credentials: bool = True
    allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])


class AuthSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    jwt_secret_key: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    api_key_enabled: bool = True
    oauth2_enabled: bool = False


class LoggingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    level: str = "info"
    format: str = "json"


class APISettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    debug: bool = False
    environment: str = "development"
    app_title: str = "IdeaFormator API"
    app_description: str = "Production-grade REST API for the IdeaFormator intelligence platform"
    app_version: str = "4.0.0"
    schema_version: str = "4.0"
    pipeline_version: str = "4.0"
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    cors: CORSSettings = Field(default_factory=CORSSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    knowledge_dir: str = "pain_intelligence/knowledge/assets/phase3"
    config_path: str = "configs/default.yaml"


def _merge_env(settings: dict[str, Any]) -> dict[str, Any]:
    env = os.environ
    if env.get("API_SECRET_KEY"):
        settings.setdefault("auth", {}).setdefault("jwt_secret_key", env["API_SECRET_KEY"])
    if env.get("JWT_SECRET_KEY"):
        settings.setdefault("auth", {}).setdefault("jwt_secret_key", env["JWT_SECRET_KEY"])
    if env.get("REDIS_URL"):
        settings.setdefault("cache", {}).setdefault("redis_url", env["REDIS_URL"])
    if env.get("CORS_ORIGINS"):
        origins = [o.strip() for o in env["CORS_ORIGINS"].split(",")]
        settings.setdefault("cors", {}).setdefault("allowed_origins", origins)
    if env.get("API_ENVIRONMENT"):
        settings["environment"] = env["API_ENVIRONMENT"]
    if env.get("API_DEBUG"):
        settings["debug"] = env["API_DEBUG"].lower() in ("true", "1", "yes")
    return settings


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("api", {})


def load_api_config(yaml_path: str | Path | None = None) -> APISettings:
    raw: dict[str, Any] = {}
    if yaml_path:
        raw = _load_yaml(yaml_path)
    raw = _merge_env(raw)
    return APISettings(**raw)
