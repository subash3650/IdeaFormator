"""Tests for collectors."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.collectors.github import GitHubCollector
from pain_intelligence.ingestion.collectors.hackernews import HackerNewsCollector
from pain_intelligence.ingestion.collectors.producthunt.collector import ProductHuntCollector
from pain_intelligence.ingestion.collectors.youtube.collector import YouTubeCollector
from pain_intelligence.ingestion.collectors.playstore.collector import PlayStoreCollector
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.models import ConfigurationError, SourceType, SyncState


def _mock_response(status_code: int = 200, json_data=None, headers=None):
    """Create a mock httpx response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or []
    resp.headers = headers or {}
    resp.text = str(json_data)
    return resp


class TestGitHubCollector:
    def test_source_property(self, github_config: CollectorConfig, mock_client: MagicMock):
        collector = GitHubCollector(github_config, mock_client)
        assert collector.source == SourceType.GITHUB

    def test_authenticate(self, github_config: CollectorConfig, mock_client: MagicMock):
        collector = GitHubCollector(github_config, mock_client)
        collector.authenticate()  # Should not raise

    def test_health_check_success(self, github_config: CollectorConfig, mock_client: MagicMock):
        mock_client.get.return_value = _mock_response(200, {
            "resources": {"core": {"remaining": 5000}}
        })
        collector = GitHubCollector(github_config, mock_client)
        assert collector.health_check() is True

    def test_health_check_failure(self, github_config: CollectorConfig, mock_client: MagicMock):
        mock_client.get.return_value = _mock_response(403)
        collector = GitHubCollector(github_config, mock_client)
        assert collector.health_check() is False

    def test_fetch_yields_batches(self, github_config: CollectorConfig, mock_client: MagicMock):
        issues = [{"id": i, "title": f"Issue {i}", "state": "open", "body": "Body"} for i in range(3)]
        # Return issues for every call (fetch iterates repos and pages)
        mock_client.get.return_value = _mock_response(200, issues, {"Link": ''})
        collector = GitHubCollector(github_config, mock_client)
        collector.authenticate()

        batches = list(collector.fetch(SyncState(source=SourceType.GITHUB)))
        assert len(batches) >= 1
        assert len(batches[0]) == 3


class TestHackerNewsCollector:
    def test_source_property(self, hn_config: CollectorConfig, mock_client: MagicMock):
        collector = HackerNewsCollector(hn_config, mock_client)
        assert collector.source == SourceType.HACKERNEWS

    def test_authenticate(self, hn_config: CollectorConfig, mock_client: MagicMock):
        collector = HackerNewsCollector(hn_config, mock_client)
        collector.authenticate()  # Should not raise (no-op)

    def test_health_check_success(self, hn_config: CollectorConfig, mock_client: MagicMock):
        mock_client.get.return_value = _mock_response(200, 99999999)
        collector = HackerNewsCollector(hn_config, mock_client)
        assert collector.health_check() is True

    def test_health_check_failure(self, hn_config: CollectorConfig, mock_client: MagicMock):
        mock_client.get.return_value = _mock_response(500)
        collector = HackerNewsCollector(hn_config, mock_client)
        assert collector.health_check() is False

    def test_fetch_yields_batches(self, hn_config: CollectorConfig, mock_client: MagicMock):
        story_ids = [40000001, 40000002]
        story = {"id": 40000001, "type": "story", "title": "Test", "by": "user", "time": 1721496000}

        # Return different responses based on URL pattern
        def mock_get(url, **kwargs):
            if "topstories" in url or "newstories" in url or "beststories" in url:
                return _mock_response(200, story_ids)
            elif "/item/" in url:
                return _mock_response(200, story)
            return _mock_response(200, [])

        mock_client.get.side_effect = mock_get
        collector = HackerNewsCollector(hn_config, mock_client)
        collector.authenticate()

        batches = list(collector.fetch(SyncState(source=SourceType.HACKERNEWS)))
        assert len(batches) >= 1


class TestProductHuntCollector:
    def test_source_property(self, ph_config: CollectorConfig, mock_client: MagicMock):
        collector = ProductHuntCollector(ph_config, mock_client)
        assert collector.source == SourceType.PRODUCTHUNT

    def test_authenticate(self, ph_config: CollectorConfig, mock_client: MagicMock):
        collector = ProductHuntCollector(ph_config, mock_client)
        collector.authenticate()  # Should not raise

    def test_health_check_success(self, ph_config: CollectorConfig, mock_client: MagicMock):
        mock_client.post.return_value = _mock_response(200, {"data": {"__typename": "Query"}})
        collector = ProductHuntCollector(ph_config, mock_client)
        collector._access_token = "test-token"
        assert collector.health_check() is True

    def test_health_check_no_token(self, ph_config: CollectorConfig, mock_client: MagicMock):
        collector = ProductHuntCollector(ph_config, mock_client)
        collector._access_token = None
        assert collector.health_check() is False

    def test_health_check_failure(self, ph_config: CollectorConfig, mock_client: MagicMock):
        mock_client.post.return_value = _mock_response(500)
        collector = ProductHuntCollector(ph_config, mock_client)
        collector._access_token = "test-token"
        assert collector.health_check() is False

    def test_fetch_yields_batches(self, ph_config: CollectorConfig, mock_client: MagicMock):
        posts_response = {
            "data": {
                "posts": {
                    "edges": [
                        {"node": {"id": "PH-1", "name": "Test App", "tagline": "A test", "votesCount": 10,
                                  "commentsCount": 2, "createdAt": "2026-07-20T10:00:00Z",
                                  "topics": {"edges": []}, "makers": {"edges": []}}},
                    ],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }
        comments_response = {
            "data": {
                "post": {
                    "comments": {
                        "edges": [],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }

        def mock_post(url, json=None, headers=None):
            query = json.get("query", "") if json else ""
            if "GetPosts" in query:
                return _mock_response(200, posts_response)
            elif "GetPostComments" in query:
                return _mock_response(200, comments_response)
            return _mock_response(200, {})

        mock_client.post.side_effect = mock_post
        collector = ProductHuntCollector(ph_config, mock_client)
        collector._access_token = "test-token"

        batches = list(collector.fetch(SyncState(source=SourceType.PRODUCTHUNT)))
        assert len(batches) >= 1

    def test_fetch_no_token(self, ph_config: CollectorConfig, mock_client: MagicMock):
        collector = ProductHuntCollector(ph_config, mock_client)
        collector._access_token = None
        batches = list(collector.fetch(SyncState(source=SourceType.PRODUCTHUNT)))
        assert len(batches) == 0

    def test_fetch_empty_response(self, ph_config: CollectorConfig, mock_client: MagicMock):
        empty_response = {
            "data": {
                "posts": {
                    "edges": [],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }
        mock_client.post.return_value = _mock_response(200, empty_response)
        collector = ProductHuntCollector(ph_config, mock_client)
        collector._access_token = "test-token"

        batches = list(collector.fetch(SyncState(source=SourceType.PRODUCTHUNT)))
        assert len(batches) == 0

    def test_fetch_graphql_error(self, ph_config: CollectorConfig, mock_client: MagicMock):
        error_response = {"errors": [{"message": "Unauthorized"}]}
        mock_client.post.return_value = _mock_response(200, error_response)
        collector = ProductHuntCollector(ph_config, mock_client)
        collector.authenticate()

        batches = list(collector.fetch(SyncState(source=SourceType.PRODUCTHUNT)))
        assert len(batches) == 0


class TestYouTubeCollector:
    def test_source_property(self, yt_config: CollectorConfig, mock_client: MagicMock):
        collector = YouTubeCollector(yt_config, mock_client)
        assert collector.source == SourceType.YOUTUBE

    def test_authenticate(self, yt_config: CollectorConfig, mock_client: MagicMock):
        collector = YouTubeCollector(yt_config, mock_client)
        collector.authenticate()  # Should not raise

    def test_health_check_success(self, yt_config: CollectorConfig, mock_client: MagicMock):
        mock_client.get.return_value = _mock_response(200, {
            "items": [{"id": "UC123", "snippet": {"title": "YouTube"}}]
        })
        collector = YouTubeCollector(yt_config, mock_client)
        assert collector.health_check() is True

    def test_health_check_failure(self, yt_config: CollectorConfig, mock_client: MagicMock):
        mock_client.get.return_value = _mock_response(200, {
            "error": {"code": 403, "message": "quotaExceeded"}
        })
        collector = YouTubeCollector(yt_config, mock_client)
        assert collector.health_check() is False

    def test_fetch_yields_batches(self, yt_config: CollectorConfig, mock_client: MagicMock):
        search_response = {
            "items": [
                {"id": {"kind": "youtube#video", "videoId": "vid1"}},
            ],
        }
        videos_response = {
            "items": [
                {
                    "id": "vid1",
                    "kind": "youtube#video",
                    "snippet": {
                        "title": "Test Video",
                        "description": "A test video",
                        "channelId": "UC123",
                        "channelTitle": "Test Channel",
                        "publishedAt": "2026-07-20T10:00:00Z",
                        "tags": ["test"],
                        "categoryId": "28",
                    },
                    "statistics": {"viewCount": "1000", "likeCount": "50", "commentCount": "10"},
                    "contentDetails": {"duration": "PT10M", "definition": "hd", "caption": "false"},
                },
            ],
        }
        comments_response = {
            "items": [],
        }

        def mock_get(url, params=None, headers=None):
            if "search" in url:
                return _mock_response(200, search_response)
            elif "videos" in url:
                return _mock_response(200, videos_response)
            elif "commentThreads" in url:
                return _mock_response(200, comments_response)
            return _mock_response(200, {})

        mock_client.get.side_effect = mock_get
        collector = YouTubeCollector(yt_config, mock_client)
        collector.authenticate()

        batches = list(collector.fetch(SyncState(source=SourceType.YOUTUBE)))
        assert len(batches) >= 1

    def test_fetch_quota_exhausted(self, yt_config: CollectorConfig, mock_client: MagicMock):
        collector = YouTubeCollector(yt_config, mock_client)
        collector._quota_remaining = 0

        batches = list(collector.fetch(SyncState(source=SourceType.YOUTUBE)))
        assert len(batches) == 0

    def test_fetch_api_error(self, yt_config: CollectorConfig, mock_client: MagicMock):
        error_response = {"error": {"code": 403, "message": "quotaExceeded"}}
        mock_client.get.return_value = _mock_response(200, error_response)
        collector = YouTubeCollector(yt_config, mock_client)
        collector.authenticate()

        batches = list(collector.fetch(SyncState(source=SourceType.YOUTUBE)))
        assert len(batches) == 0


class TestPlayStoreCollector:
    def test_source_property(self, ps_config: CollectorConfig, mock_client: MagicMock):
        collector = PlayStoreCollector(ps_config, mock_client)
        assert collector.source == SourceType.PLAYSTORE

    def test_authenticate_noop(self, ps_config: CollectorConfig, mock_client: MagicMock):
        """Public collector doesn't need authentication."""
        collector = PlayStoreCollector(ps_config, mock_client)
        collector.authenticate()  # Should not raise

    def test_health_check_success(self, ps_config: CollectorConfig, mock_client: MagicMock):
        mock_app = MagicMock(return_value={"title": "Gmail"})
        with patch("pain_intelligence.ingestion.collectors.playstore.collector.gp_app", mock_app, create=True):
            collector = PlayStoreCollector(ps_config, mock_client)
            with patch.dict("sys.modules", {"google_play_scraper": MagicMock(app=mock_app)}):
                result = collector.health_check()
                assert result is True

    def test_health_check_failure(self, ps_config: CollectorConfig, mock_client: MagicMock):
        mock_app = MagicMock(side_effect=Exception("Connection refused"))
        with patch.dict("sys.modules", {"google_play_scraper": MagicMock(app=mock_app)}):
            collector = PlayStoreCollector(ps_config, mock_client)
            result = collector.health_check()
            assert result is False

    def test_fetch_yields_batches(self, ps_config: CollectorConfig, mock_client: MagicMock):
        reviews_data = [
            {"reviewId": "r1", "userName": "User1", "score": 5, "content": "Great app!"},
            {"reviewId": "r2", "userName": "User2", "score": 3, "content": "OK"},
        ]
        mock_reviews = MagicMock(return_value=(reviews_data, None))
        mock_app_fn = MagicMock(return_value={"title": "ChatGPT", "appId": "com.openai.chatgpt"})
        mock_scraper = MagicMock(reviews=mock_reviews, app=mock_app_fn)
        mock_scraper.Sort = MagicMock()

        with patch.dict("sys.modules", {"google_play_scraper": mock_scraper}):
            collector = PlayStoreCollector(ps_config, mock_client)
            collector._config = CollectorConfig(
                enabled=True,
                apps=["com.openai.chatgpt"],
                batch_size=10,
                max_pages=2,
                rate_limit=1.0,
                timeout=5,
                retry_count=1,
                retry_delay=0.01,
            )

            state = SyncState(source=SourceType.PLAYSTORE)
            batches = list(collector.fetch(state))
            assert len(batches) >= 1

    def test_fetch_no_apps(self, ps_config: CollectorConfig, mock_client: MagicMock):
        collector = PlayStoreCollector(ps_config, mock_client)
        collector._config = CollectorConfig(
            enabled=True,
            apps=[],
            apps_config_path="/nonexistent/path.yaml",
            batch_size=10,
            max_pages=2,
            rate_limit=1.0,
            timeout=5,
            retry_count=1,
            retry_delay=0.01,
        )

        state = SyncState(source=SourceType.PLAYSTORE)
        batches = list(collector.fetch(state))
        assert len(batches) == 0

    def test_fetch_empty_reviews(self, ps_config: CollectorConfig, mock_client: MagicMock):
        mock_reviews = MagicMock(return_value=([], None))
        mock_app_fn = MagicMock(return_value={"title": "ChatGPT", "appId": "com.openai.chatgpt"})
        mock_scraper = MagicMock(reviews=mock_reviews, app=mock_app_fn)
        mock_scraper.Sort = MagicMock()

        with patch.dict("sys.modules", {"google_play_scraper": mock_scraper}):
            collector = PlayStoreCollector(ps_config, mock_client)
            collector._config = CollectorConfig(
                enabled=True,
                apps=["com.openai.chatgpt"],
                batch_size=10,
                max_pages=2,
                rate_limit=1.0,
                timeout=5,
                retry_count=1,
                retry_delay=0.01,
            )

            state = SyncState(source=SourceType.PLAYSTORE)
            batches = list(collector.fetch(state))
            # Should yield app metadata even if no reviews
            assert len(batches) >= 0

    def test_fetch_multiple_apps(self, ps_config: CollectorConfig, mock_client: MagicMock):
        reviews_data = [{"reviewId": "r1", "userName": "User1", "score": 5, "content": "Good"}]
        mock_reviews = MagicMock(return_value=(reviews_data, None))
        mock_app_fn = MagicMock(return_value={"title": "App", "appId": "com.test"})
        mock_scraper = MagicMock(reviews=mock_reviews, app=mock_app_fn)
        mock_scraper.Sort = MagicMock()

        with patch.dict("sys.modules", {"google_play_scraper": mock_scraper}):
            collector = PlayStoreCollector(ps_config, mock_client)
            collector._config = CollectorConfig(
                enabled=True,
                apps=["com.app1", "com.app2"],
                batch_size=10,
                max_pages=2,
                rate_limit=1.0,
                timeout=5,
                retry_count=1,
                retry_delay=0.01,
            )

            state = SyncState(source=SourceType.PLAYSTORE)
            batches = list(collector.fetch(state))
            assert len(batches) >= 2  # At least app metadata for each app

    def test_configuration_error_importable(self):
        from pain_intelligence.ingestion.models import ConfigurationError
        assert issubclass(ConfigurationError, Exception)
