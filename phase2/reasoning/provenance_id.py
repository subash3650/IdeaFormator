"""Provenance ID generation utilities."""

from __future__ import annotations

from datetime import datetime, timezone


def generate_run_id() -> str:
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H%M%S-%f")[:21]
