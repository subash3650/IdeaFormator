"""Shared test fixtures for ingestion framework tests."""

from __future__ import annotations

import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from loguru import logger as _loguru_logger

from pain_intelligence.ingestion.adapters.github import GitHubAdapter
from pain_intelligence.ingestion.adapters.hackernews import HackerNewsAdapter
from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.collectors.producthunt.adapter import ProductHuntAdapter
from pain_intelligence.ingestion.collectors.youtube.adapter import YouTubeAdapter
from pain_intelligence.ingestion.collectors.playstore.adapter import PlayStoreAdapter
from pain_intelligence.ingestion.config import CollectorConfig, IngestionConfig
from pain_intelligence.ingestion.models import RawDocument, SourceType


@pytest.fixture(autouse=True)
def _cleanup_loguru():
    """Remove all loguru handlers after each test to prevent file locks."""
    yield
    _loguru_logger.remove()


@pytest.fixture
def tmp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_client() -> MagicMock:
    """A mock HTTP client that returns configurable responses."""
    client = MagicMock(spec=HttpClient)
    return client


@pytest.fixture
def github_config() -> CollectorConfig:
    """GitHub collector configuration for testing."""
    return CollectorConfig(
        enabled=True,
        api_key_env=None,
        batch_size=5,
        max_pages=2,
        rate_limit=10.0,
        timeout=5,
        retry_count=1,
        retry_delay=0.01,
    )


@pytest.fixture
def hn_config() -> CollectorConfig:
    """HackerNews collector configuration for testing."""
    return CollectorConfig(
        enabled=True,
        api_key_env=None,
        batch_size=10,
        max_pages=2,
        rate_limit=10.0,
        timeout=5,
        retry_count=1,
        retry_delay=0.01,
    )


@pytest.fixture
def ph_config() -> CollectorConfig:
    """Product Hunt collector configuration for testing."""
    return CollectorConfig(
        enabled=True,
        api_key_env=None,
        batch_size=10,
        max_pages=2,
        rate_limit=5.0,
        timeout=5,
        retry_count=1,
        retry_delay=0.01,
    )


@pytest.fixture
def yt_config() -> CollectorConfig:
    """YouTube collector configuration for testing."""
    return CollectorConfig(
        enabled=True,
        api_key_env=None,
        batch_size=10,
        max_pages=2,
        rate_limit=2.0,
        timeout=5,
        retry_count=1,
        retry_delay=0.01,
    )


@pytest.fixture
def ps_config() -> CollectorConfig:
    """Play Store collector configuration for testing."""
    return CollectorConfig(
        enabled=True,
        api_key_env=None,
        batch_size=10,
        max_pages=2,
        rate_limit=2.0,
        timeout=5,
        retry_count=1,
        retry_delay=0.01,
    )


@pytest.fixture
def ingestion_config() -> IngestionConfig:
    """Full ingestion configuration for testing."""
    return IngestionConfig(
        pipeline_version="0.1.0-test",
        schedule="once",
        collectors={
            "github": CollectorConfig(enabled=True, batch_size=5),
            "hackernews": CollectorConfig(enabled=True, batch_size=10),
        },
    )


@pytest.fixture
def sample_github_issue() -> dict[str, Any]:
    """A sample GitHub issue API response."""
    return {
        "id": 123456789,
        "number": 42,
        "title": "Bug: Application crashes on startup",
        "body": "When I start the application, it crashes with a segfault. This happens on Linux and macOS.",
        "state": "open",
        "created_at": "2026-07-20T10:30:00Z",
        "updated_at": "2026-07-22T14:20:00Z",
        "user": {"login": "testuser"},
        "labels": [{"name": "bug"}, {"name": "critical"}],
        "comments": 5,
        "html_url": "https://github.com/test/repo/issues/42",
        "repository_url": "https://api.github.com/repos/test/repo",
        "author_association": "MEMBER",
        "locked": False,
        "assignees": [{"login": "dev1"}],
        "milestone": {"title": "v2.0"},
        "reactions": {"total_count": 10},
    }


@pytest.fixture
def sample_github_comment() -> dict[str, Any]:
    """A sample GitHub issue comment API response."""
    return {
        "id": 987654321,
        "body": "I can reproduce this on Ubuntu 22.04. Here's the stack trace...",
        "created_at": "2026-07-21T08:15:00Z",
        "updated_at": "2026-07-21T08:15:00Z",
        "user": {"login": "contributor1"},
        "html_url": "https://github.com/test/repo/issues/42#issuecomment-987654321",
        "issue_url": "https://api.github.com/repos/test/repo/issues/42",
        "author_association": "CONTRIBUTOR",
        "reactions": {"total_count": 3},
    }


@pytest.fixture
def sample_hn_story() -> dict[str, Any]:
    """A sample HN story API response."""
    return {
        "id": 40000001,
        "type": "story",
        "title": "Show HN: I built a tool for real-time code collaboration",
        "url": "https://example.com/collab-tool",
        "text": "",
        "by": "hnuser1",
        "time": 1721496000,  # 2024-07-21T12:00:00Z
        "score": 150,
        "descendants": 45,
        "kids": [40000002, 40000003],
        "dead": False,
        "deleted": False,
    }


@pytest.fixture
def sample_hn_comment() -> dict[str, Any]:
    """A sample HN comment API response."""
    return {
        "id": 40000002,
        "type": "comment",
        "text": "This is really interesting. I've been looking for something like this for a while.",
        "by": "commenter1",
        "time": 1721499600,
        "parent": 40000001,
        "kids": [],
        "dead": False,
        "deleted": False,
    }


@pytest.fixture
def sample_hn_askhn() -> dict[str, Any]:
    """A sample Ask HN story."""
    return {
        "id": 40000010,
        "type": "story",
        "title": "Ask HN: What's the best way to learn distributed systems?",
        "url": "",
        "text": "I'm a mid-level developer looking to level up my distributed systems knowledge. What resources do you recommend?",
        "by": "asker1",
        "time": 1721500000,
        "score": 200,
        "descendants": 80,
        "kids": [40000011],
        "dead": False,
        "deleted": False,
    }


@pytest.fixture
def sample_ph_post() -> dict[str, Any]:
    """A sample Product Hunt post GraphQL node."""
    return {
        "id": "PH-12345",
        "name": "AI Code Review Bot",
        "tagline": "Automated code reviews powered by GPT-4",
        "description": "An AI-powered tool that automatically reviews pull requests and provides actionable feedback on code quality, security, and performance.",
        "url": "https://www.producthunt.com/posts/ai-code-review-bot",
        "website": "https://aicodebot.example.com",
        "votesCount": 342,
        "commentsCount": 67,
        "createdAt": "2026-07-15T08:00:00Z",
        "thumbnail": {"url": "https://ph-uploads.s3.amazonaws.com/thumb_12345.jpg"},
        "topics": {
            "edges": [
                {"node": {"name": "Developer Tools"}},
                {"node": {"name": "Artificial Intelligence"}},
            ]
        },
        "makers": {
            "edges": [
                {"node": {"id": "user-001", "name": "Jane Developer", "username": "janedev"}}
            ]
        },
    }


@pytest.fixture
def sample_ph_comment() -> dict[str, Any]:
    """A sample Product Hunt comment GraphQL node."""
    return {
        "id": "PH-C-001",
        "commentBody": "This is amazing! I've been using it for a week and it catches bugs I would have missed.",
        "createdAt": "2026-07-16T10:30:00Z",
        "author": {
            "name": "Bob Reviewer",
            "username": "bobreviewer",
        },
        "post": {
            "id": "PH-12345",
        },
    }


@pytest.fixture
def sample_yt_video() -> dict[str, Any]:
    """A sample YouTube video API response."""
    return {
        "kind": "youtube#video",
        "id": "dQw4w9WgXcQ",
        "snippet": {
            "publishedAt": "2026-07-15T12:00:00Z",
            "channelId": "UC1234567890",
            "title": "Building a Startup in 2026: Complete Guide",
            "description": "In this video, we cover everything you need to know about building a successful startup in 2026.",
            "tags": ["startup", "entrepreneurship", "2026"],
            "categoryId": "28",
            "channelTitle": "Tech Startup Hub",
            "defaultLanguage": "en",
        },
        "statistics": {
            "viewCount": "125000",
            "likeCount": "4500",
            "commentCount": "342",
        },
        "contentDetails": {
            "duration": "PT25M30S",
            "definition": "hd",
            "caption": "true",
        },
    }


@pytest.fixture
def sample_yt_comment_thread() -> dict[str, Any]:
    """A sample YouTube comment thread API response."""
    return {
        "kind": "youtube#commentThread",
        "id": "comment-thread-001",
        "snippet": {
            "videoId": "dQw4w9WgXcQ",
            "topLevelComment": {
                "id": "yt-comment-001",
                "snippet": {
                    "textDisplay": "Great video! Very informative and well-structured.",
                    "textOriginal": "Great video! Very informative and well-structured.",
                    "authorDisplayName": "TechViewer42",
                    "authorChannelId": {"value": "UC9999999999"},
                    "publishedAt": "2026-07-16T08:00:00Z",
                    "updatedAt": "2026-07-16T08:00:00Z",
                    "likeCount": 15,
                },
            },
            "totalReplyCount": 3,
        },
    }


@pytest.fixture
def sample_ps_review() -> dict[str, Any]:
    """A sample Google Play review from google-play-scraper."""
    return {
        "reviewId": "abc123-def456-ghi789",
        "userName": "App Reviewer 42",
        "userImage": "https://play-lh.googleusercontent.com/a-/AOh14Gh_test",
        "content": "Great app! Works well on my Pixel 8. The new update fixed most of the issues.",
        "score": 4,
        "thumbsUpCount": 5,
        "reviewCreatedVersion": "1.32.0",
        "at": datetime(2026, 7, 15, 14, 30, 0),
        "replyContent": None,
        "repliedAt": None,
        "appVersion": "1.32.0",
    }


@pytest.fixture
def sample_ps_review_with_reply() -> dict[str, Any]:
    """A sample Google Play review with developer reply."""
    return {
        "reviewId": "xyz789-abc123-def456",
        "userName": "Beta Tester",
        "userImage": "https://play-lh.googleusercontent.com/a-/AOh14Gh_test2",
        "content": "App keeps crashing on startup after the latest update.",
        "score": 2,
        "thumbsUpCount": 12,
        "reviewCreatedVersion": "5.2.0",
        "at": datetime(2026, 7, 14, 10, 0, 0),
        "replyContent": "Thank you for reporting. We've fixed this in version 5.2.1.",
        "repliedAt": datetime(2026, 7, 15, 12, 0, 0),
        "appVersion": "5.2.0",
    }


@pytest.fixture
def github_adapter() -> GitHubAdapter:
    """A GitHubAdapter instance."""
    return GitHubAdapter()


@pytest.fixture
def hn_adapter() -> HackerNewsAdapter:
    """A HackerNewsAdapter instance."""
    return HackerNewsAdapter()


@pytest.fixture
def ph_adapter() -> ProductHuntAdapter:
    """A ProductHuntAdapter instance."""
    return ProductHuntAdapter()


@pytest.fixture
def yt_adapter() -> YouTubeAdapter:
    """A YouTubeAdapter instance."""
    return YouTubeAdapter()


@pytest.fixture
def ps_adapter() -> PlayStoreAdapter:
    """A PlayStoreAdapter instance."""
    return PlayStoreAdapter()
