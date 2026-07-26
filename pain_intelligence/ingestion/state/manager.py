"""State manager for incremental sync tracking."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pain_intelligence.ingestion.models import SourceType, SyncState
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class StateManager:
    """Manages sync state for each collector.

    State is persisted as JSON files in {output_base}/state/{source}.json.
    Tracks:
    - Last sync timestamp (for incremental sync)
    - Pagination cursor / next_page_token
    - ETag (for HTTP conditional requests)
    - Failure count and last error
    - Total documents collected
    """

    def __init__(self, output_base: Path) -> None:
        self._state_dir = output_base / "state"
        self._state_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self, source: SourceType) -> SyncState:
        """Load the sync state for a source, returning a default if not found."""
        path = self._state_dir / f"{source.value}.json"
        if not path.exists():
            return SyncState(source=source)

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return SyncState(**data)
        except Exception as e:
            logger.warning("Failed to load state for {}: {}", source.value, e)
            return SyncState(source=source)

    def save_state(self, state: SyncState) -> None:
        """Persist sync state to disk."""
        path = self._state_dir / f"{state.source.value}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state.model_dump(mode="json"), f, indent=2, ensure_ascii=False, default=str)
        logger.debug("Saved state for {}", state.source.value)

    def record_success(self, source: SourceType, count: int, page_token: str | None = None) -> None:
        """Record a successful sync."""
        state = self.load_state(source)
        new_state = state.model_copy(update={
            "last_sync": datetime.now(timezone.utc),
            "total_collected": state.total_collected + count,
            "failure_count": 0,
            "last_error": None,
            "next_page_token": page_token or state.next_page_token,
        })
        self.save_state(new_state)

    def record_failure(self, source: SourceType, error: str) -> None:
        """Record a sync failure."""
        state = self.load_state(source)
        new_state = state.model_copy(update={
            "failure_count": state.failure_count + 1,
            "last_error": error,
        })
        self.save_state(new_state)

    def reset_state(self, source: SourceType) -> None:
        """Reset state for a source to defaults."""
        state = SyncState(source=source)
        self.save_state(state)
        logger.info("Reset state for {}", source.value)

    def clear_all(self) -> None:
        """Remove all state files."""
        for path in self._state_dir.glob("*.json"):
            path.unlink()
        logger.info("Cleared all state files")
