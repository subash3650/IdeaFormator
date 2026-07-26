from __future__ import annotations

from fastapi import Depends

from phase5.api.auth.models import UserContext
from phase5.api.dependencies.auth import get_current_user
from phase5.api.exceptions.base import AuthorizationError

PERMISSION_MATRIX: dict[str, list[str]] = {
    "anonymous": ["health:read", "system:read"],
    "read_only": ["health:read", "system:read", "kg:read", "reasoning:read",
                  "opportunity:read", "trend:read", "report:read",
                  "search:read", "statistics:read"],
    "read_write": ["*:read", "report:write", "copilot:write", "export:write"],
    "admin": ["*"],
    "service": ["*", "pipeline:write", "config:write"],
}


async def require_permission(resource: str, action: str, user: UserContext = Depends(get_current_user)) -> UserContext:
    for role in user.roles:
        perms = PERMISSION_MATRIX.get(role, [])
        if "*" in perms:
            return user
        if f"{resource}:{action}" in perms:
            return user
        if f"{resource}:read" in perms and action == "read":
            return user
        if f"*:{action}" in perms:
            return user
    raise AuthorizationError(details={"required": f"{resource}:{action}", "user_roles": user.roles})
