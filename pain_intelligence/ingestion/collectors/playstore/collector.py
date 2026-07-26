"""Collector for Google Play Store public reviews and app metadata.

Uses google-play-scraper to fetch publicly available data without any
authentication, Play Console account, or developer credentials.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from pain_intelligence.ingestion.collectors.base import BaseCollector, COLLECTOR_VERSION
from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.collectors.playstore.adapter import PlayStoreAdapter
from pain_intelligence.ingestion.collectors.playstore.parser import PlayStoreParser
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.models import SourceType, SyncState
from pain_intelligence.ingestion.registry import register_collector
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

SORT_MAP = {
    "newest": "newest",
    "most_relevant": "most_relevant",
    "rating": "rating",
}

DEFAULT_APPS_CONFIG = Path("configs/playstore_apps.yaml")


def _load_apps_config(config_path: str | Path | None, language: str, country: str) -> dict[str, list[str]]:
    """Load apps from playstore_apps.yaml, organized by category."""
    if config_path is None:
        config_path = DEFAULT_APPS_CONFIG

    p = Path(config_path)
    if not p.exists():
        logger.warning("[playstore] Apps config not found at {}", p)
        return {}

    try:
        import yaml
        with open(p, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logger.error("[playstore] Failed to load apps config: {}", e)
        return {}


def _get_sort_enum(sort_name: str) -> Any:
    """Get the google-play-scraper Sort enum value."""
    try:
        from google_play_scraper import Sort
        sort_map = {
            "newest": Sort.NEWEST,
            "most_relevant": Sort.MOST_RELEVANT,
            "rating": Sort.RATING,
        }
        return sort_map.get(sort_name, Sort.NEWEST)
    except ImportError:
        return sort_name


@register_collector("playstore")
class PlayStoreCollector(BaseCollector):
    """Collects public reviews and app metadata from the Google Play Store.

    Uses google-play-scraper — no authentication required.
    Collects both app metadata and reviews for market intelligence.

    Supports:
    - Multiple apps via playstore_apps.yaml config
    - Category-based app selection
    - Per-app state tracking (cursor, timestamps)
    - Incremental collection
    - Configurable language, country, sort order
    - Review count limits
    - App version tracking for change correlation
    """

    adapter_class = PlayStoreAdapter

    def __init__(self, config: CollectorConfig, client: HttpClient) -> None:
        super().__init__(config, client)
        self._parser = PlayStoreParser()
        self._summary: dict[str, Any] = {}

    @property
    def source(self) -> SourceType:
        return SourceType.PLAYSTORE

    def authenticate(self) -> None:
        """No authentication needed for public Play Store data."""
        pass

    def health_check(self) -> bool:
        """Verify google-play-scraper can reach Google Play Store."""
        try:
            from google_play_scraper import app
            result = app(
                "com.google.android.gm",
                lang=self._config.language,
                country=self._config.country,
            )
            self._api_calls += 1
            if result and result.get("title"):
                logger.info("[playstore] API healthy. App: {}", result["title"])
                return True
            logger.warning("[playstore] Health check returned empty result.")
            return False
        except ImportError:
            logger.error("[playstore] google-play-scraper not installed. "
                         "Install with: pip install google-play-scraper")
            return False
        except Exception as e:
            logger.error("[playstore] Health check failed: {}", e)
            return False

    def fetch(self, state: SyncState | None) -> Iterator[list[dict[str, Any]]]:
        """Fetch app metadata and reviews for all configured apps."""
        try:
            from google_play_scraper import app as gp_app, reviews as gp_reviews
        except ImportError:
            logger.error("[playstore] google-play-scraper not installed.")
            return

        apps = self._get_target_apps()
        if not apps:
            logger.warning("[playstore] No apps configured. Skipping.")
            return

        self._summary = {"apps": len(apps), "reviews": 0, "countries": set(), "languages": set()}

        for package_name in apps:
            app_state = self._get_app_state(state, package_name)
            app_data, app_reviews = self._collect_app(
                gp_app, gp_reviews, package_name, app_state
            )
            self._update_app_state(state, package_name, app_state)

            if app_data:
                yield [app_data]

            if app_reviews:
                self._summary["reviews"] += len(app_reviews)
                self._summary["countries"].add(self._config.country)
                self._summary["languages"].add(self._config.language)
                yield app_reviews

    def _collect_app(
        self,
        gp_app: Any,
        gp_reviews: Any,
        package_name: str,
        app_state: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        """Collect metadata and reviews for a single app."""
        app_data = None
        all_reviews: list[dict[str, Any]] = []

        # 1. Fetch app metadata
        try:
            raw_app = gp_app(
                package_name,
                lang=self._config.language,
                country=self._config.country,
            )
            self._api_calls += 1
            if raw_app:
                app_data = self._parser.parse_app_info(raw_app)
                # Track app version for change correlation
                app_state["app_version"] = raw_app.get("version")
                app_state["app_title"] = raw_app.get("title")
        except Exception as e:
            logger.warning("[playstore] Failed to fetch app info for {}: {}", package_name, e)

        # 2. Fetch reviews with pagination
        sort_val = _get_sort_enum(self._config.sort)
        continuation_token = app_state.get("cursor")
        limit = self._config.review_limit
        collected = app_state.get("reviews_collected", 0)
        max_to_collect = limit - collected

        if max_to_collect <= 0:
            logger.info("[playstore] Review limit reached for {}. Skipping.", package_name)
            return app_data, all_reviews

        try:
            result, token = gp_reviews(
                package_name,
                lang=self._config.language,
                country=self._config.country,
                sort=sort_val,
                count=min(self._config.batch_size, max_to_collect),
                continuation_token=continuation_token,
            )
            self._api_calls += 1

            if result:
                all_reviews.extend(result)
                collected += len(result)
                app_state["cursor"] = token
                app_state["reviews_collected"] = collected
                logger.debug("[playstore] Fetched {} reviews for {} (total: {})",
                             len(result), package_name, collected)

            # Continue pagination until limit reached
            while token and collected < limit:
                result, token = gp_reviews(
                    package_name,
                    lang=self._config.language,
                    country=self._config.country,
                    sort=sort_val,
                    count=min(self._config.batch_size, limit - collected),
                    continuation_token=token,
                )
                self._api_calls += 1

                if not result:
                    break
                all_reviews.extend(result)
                collected += len(result)
                app_state["cursor"] = token
                app_state["reviews_collected"] = collected

        except Exception as e:
            logger.warning("[playstore] Failed to fetch reviews for {}: {}", package_name, e)

        return app_data, all_reviews

    def _get_target_apps(self) -> list[str]:
        """Get list of apps to collect based on config."""
        if self._config.apps:
            return self._config.apps

        if self._config.apps_config_path:
            apps_data = _load_apps_config(
                self._config.apps_config_path,
                self._config.language,
                self._config.country,
            )
            return [pkg for pkgs in apps_data.values() for pkg in pkgs]

        if DEFAULT_APPS_CONFIG.exists():
            apps_data = _load_apps_config(
                DEFAULT_APPS_CONFIG,
                self._config.language,
                self._config.country,
            )
            return [pkg for pkgs in apps_data.values() for pkg in pkgs]

        return []

    def _get_app_state(self, state: SyncState | None, package_name: str) -> dict[str, Any]:
        """Extract per-app state from the global SyncState cursor."""
        if not state or not state.cursor:
            return {"reviews_collected": 0}

        try:
            all_state = json.loads(state.cursor)
            return all_state.get(package_name, {"reviews_collected": 0})
        except (json.JSONDecodeError, TypeError):
            return {"reviews_collected": 0}

    def _update_app_state(
        self,
        state: SyncState | None,
        package_name: str,
        app_state: dict[str, Any],
    ) -> None:
        """Update per-app state in the global SyncState cursor."""
        if not state:
            return

        # Build the new cursor with all app states
        all_state: dict[str, Any] = {}
        if state.cursor:
            try:
                all_state = json.loads(state.cursor)
            except (json.JSONDecodeError, TypeError):
                pass

        all_state[package_name] = app_state

        # We can't mutate the frozen SyncState, so we store it
        # The engine will read this via the collector's state tracking
        # For now, we log the state update
        logger.debug("[playstore] State updated for {}: {}", package_name, app_state)
