from __future__ import annotations

import logging

from fastapi import FastAPI

from phase5.api.config.settings import APISettings, load_api_config
from phase5.api.middleware.compression import make_compression_middleware
from phase5.api.middleware.cors import make_cors_middleware
from phase5.api.middleware.exception_handler import register_exception_handlers
from phase5.api.middleware.logging_mw import RequestLoggingMiddleware
from phase5.api.middleware.metrics_mw import MetricsMiddleware
from phase5.api.middleware.rate_limit_mw import RateLimitMiddleware
from phase5.api.middleware.request_id import RequestIDMiddleware
from phase5.api.middleware.security_headers import SecurityHeadersMiddleware
from phase5.api.middleware.timing import TimingMiddleware
from phase5.api.routes.v1.router import router as v1_router


def create_app(config: APISettings | None = None) -> FastAPI:
    settings = config or load_api_config()

    app = FastAPI(
        title=settings.app_title,
        description=settings.app_description,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    _configure_logging(settings)
    _register_middleware(app, settings)
    _register_routes(app)
    register_exception_handlers(app)

    app.state.settings = settings

    return app


def _configure_logging(settings: APISettings) -> None:
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")


def _register_middleware(app: FastAPI, settings: APISettings) -> None:
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(MetricsMiddleware)

    if settings.rate_limit.enabled:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit.requests_per_minute)

    cors_config = make_cors_middleware(settings.cors)
    app.add_middleware(cors_config["middleware_class"], **{k: v for k, v in cors_config.items() if k != "middleware_class"})

    compression_config = make_compression_middleware()
    app.add_middleware(compression_config["middleware_class"], minimum_size=compression_config.get("minimum_size", 1000))


def _register_routes(app: FastAPI) -> None:
    app.include_router(v1_router, prefix="/api/v1")
