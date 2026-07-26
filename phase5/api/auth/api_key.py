from __future__ import annotations

import secrets


def generate_api_key(prefix: str = "ak") -> str:
    random_part = secrets.token_hex(32)
    return f"{prefix}_{random_part}"
