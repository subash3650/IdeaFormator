from __future__ import annotations

from starlette.middleware.cors import CORSMiddleware

from phase5.api.config.settings import CORSSettings


def make_cors_middleware(settings: CORSSettings) -> dict:
    return {
        "middleware_class": CORSMiddleware,
        "allow_origins": settings.allowed_origins,
        "allow_credentials": settings.allow_credentials,
        "allow_methods": settings.allow_methods,
        "allow_headers": settings.allow_headers,
    }
