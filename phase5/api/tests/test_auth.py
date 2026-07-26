from __future__ import annotations

from phase5.api.auth.api_key import generate_api_key
from phase5.api.auth.models import UserContext, TokenPayload
from phase5.api.auth.service import AuthService


class TestAuthService:
    def setup_method(self):
        self.service = AuthService(secret_key="test-secret")

    def test_validate_api_key_valid(self):
        key = generate_api_key()
        user = self.service.validate_api_key(key)
        assert user is not None
        assert user.token_type == "api_key"
        assert "read" in user.permissions

    def test_validate_api_key_invalid(self):
        user = self.service.validate_api_key("invalid")
        assert user is None

    def test_validate_api_key_bad_format(self):
        user = self.service.validate_api_key("no_prefix")
        assert user is None

    def test_create_jwt(self):
        token = self.service.create_jwt("user1", ["admin"], ["read", "write"])
        assert isinstance(token, str)
        assert len(token) > 20

    def test_validate_jwt_valid(self):
        token = self.service.create_jwt("user1", ["admin"], ["read", "write"])
        user = self.service.validate_jwt(token)
        assert user.user_id == "user1"
        assert "admin" in user.roles
        assert "read" in user.permissions

    def test_validate_jwt_invalid(self):
        try:
            self.service.validate_jwt("invalid-token")
            assert False
        except Exception:
            pass

    def test_hash_api_key(self):
        key = generate_api_key()
        h1 = self.service.hash_api_key(key)
        h2 = self.service.hash_api_key(key)
        assert h1 == h2
        assert len(h1) == 64


class TestUserContext:
    def test_default_anonymous(self):
        ctx = UserContext()
        assert ctx.user_id == "anonymous"
        assert ctx.roles == ["anonymous"]
