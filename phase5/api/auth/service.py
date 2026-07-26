from __future__ import annotations

import hashlib
import hmac
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from phase5.api.auth.models import TokenPayload, UserContext
from phase5.api.exceptions.base import AuthenticationError


class AuthService:
    def __init__(self, secret_key: str = "change-me") -> None:
        self._secret_key = secret_key

    def validate_api_key(self, key: str) -> UserContext | None:
        if not key.startswith("ak_"):
            return None
        return UserContext(
            user_id="apikey_user",
            roles=["read_only"],
            permissions=["read"],
            token_type="api_key",
        )

    def create_jwt(self, user_id: str, roles: list[str], permissions: list[str], expiry_minutes: int = 60) -> str:
        import jwt
        payload = {
            "sub": user_id,
            "roles": roles,
            "permissions": permissions,
            "iat": int(time.time()),
            "exp": int(time.time()) + expiry_minutes * 60,
        }
        return jwt.encode(payload, self._secret_key, algorithm="HS256")

    def validate_jwt(self, token: str) -> UserContext:
        import jwt
        try:
            payload = jwt.decode(token, self._secret_key, algorithms=["HS256"])
            return UserContext(
                user_id=payload.get("sub", "unknown"),
                roles=payload.get("roles", []),
                permissions=payload.get("permissions", []),
                token_type="jwt",
            )
        except jwt.ExpiredSignatureError:
            raise AuthenticationError(message="Token expired")
        except jwt.InvalidTokenError:
            raise AuthenticationError(message="Invalid token")

    def hash_api_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
