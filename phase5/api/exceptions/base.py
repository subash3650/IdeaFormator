from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCodeEnum(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    VALIDATION = "VALIDATION"
    AUTHENTICATION = "AUTHENTICATION"
    AUTHORIZATION = "AUTHORIZATION"
    RATE_LIMIT = "RATE_LIMIT"
    CONFLICT = "CONFLICT"
    INTERNAL = "INTERNAL"
    BAD_REQUEST = "BAD_REQUEST"
    SERVICE_UNAVAILABLE = "SERVICE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"


class APIError(Exception):
    def __init__(
        self,
        code: ErrorCodeEnum,
        message: str = "",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message or code.value.replace("_", " ").title()
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class NotFoundError(APIError):
    def __init__(self, message: str = "Resource not found", details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCodeEnum.NOT_FOUND, message, 404, details)


class ValidationError_(APIError):
    def __init__(self, message: str = "Validation failed", details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCodeEnum.VALIDATION, message, 422, details)


class AuthenticationError(APIError):
    def __init__(self, message: str = "Authentication required", details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCodeEnum.AUTHENTICATION, message, 401, details)


class AuthorizationError(APIError):
    def __init__(self, message: str = "Insufficient permissions", details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCodeEnum.AUTHORIZATION, message, 403, details)


class RateLimitError(APIError):
    def __init__(self, message: str = "Rate limit exceeded", details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCodeEnum.RATE_LIMIT, message, 429, details)


class ConflictError(APIError):
    def __init__(self, message: str = "Resource conflict", details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCodeEnum.CONFLICT, message, 409, details)


class InternalError(APIError):
    def __init__(self, message: str = "Internal server error", details: dict[str, Any] | None = None) -> None:
        super().__init__(ErrorCodeEnum.INTERNAL, message, 500, details)
