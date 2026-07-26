from __future__ import annotations

from fastapi import Depends, Header, Request
from pydantic import BaseModel, ConfigDict

from phase5.api.exceptions.base import AuthenticationError, AuthorizationError


class UserContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    user_id: str = "anonymous"
    roles: list[str] = ["anonymous"]
    permissions: list[str] = []


async def get_current_user(
    request: Request,
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> UserContext:
    if x_api_key:
        if _validate_api_key(x_api_key):
            return UserContext(user_id="api_key_user", roles=["read_only"], permissions=["read"])
    if authorization:
        token = authorization.replace("Bearer ", "")
        user = _validate_jwt(token)
        if user:
            return user
    settings = getattr(request.app.state, "settings", None)
    if settings and settings.environment == "development":
        return UserContext(user_id="dev_user", roles=["admin"], permissions=["read", "write", "admin"])
    raise AuthenticationError(details={"header": "Authorization or X-API-Key required"})


async def require_admin(user: UserContext = Depends(get_current_user)) -> UserContext:
    if "admin" not in user.roles:
        raise AuthorizationError(details={"required_role": "admin"})
    return user


async def require_read(user: UserContext = Depends(get_current_user)) -> UserContext:
    if "read" not in user.permissions and "anonymous" in user.roles:
        raise AuthorizationError(details={"required": "read permission"})
    return user


def _validate_api_key(key: str) -> bool:
    if key.startswith("ak_"):
        return True
    return False


def _validate_jwt(token: str) -> UserContext | None:
    return None
