"""Tests for adapters with golden dataset testing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pain_intelligence.ingestion.adapters.github import GitHubAdapter
from pain_intelligence.ingestion.adapters.hackernews import HackerNewsAdapter
from pain_intelligence.ingestion.collectors.producthunt.adapter import ProductHuntAdapter
from pain_intelligence.ingestion.collectors.producthunt.parser import ProductHuntParser
from pain_intelligence.ingestion.collectors.youtube.adapter import YouTubeAdapter
from pain_intelligence.ingestion.collectors.youtube.parser import YouTubeParser
from pain_intelligence.ingestion.collectors.playstore.adapter import PlayStoreAdapter
from pain_intelligence.ingestion.collectors.playstore.parser import PlayStoreParser
from pain_intelligence.ingestion.models import SourceType


GOLDEN_DIR = Path(__file__).parent / "golden"


def _load_golden(name: str) -> dict[str, Any]:
    """Load a golden dataset JSON file."""
    with open(GOLDEN_DIR / name, encoding="utf-8") as f:
        return json.load(f)


class TestGitHubAdapter:
    def test_transform_issue(self, sample_github_issue: dict[str, Any]):
        adapter = GitHubAdapter()
        result = adapter.transform(sample_github_issue)

        assert result["source"] == SourceType.GITHUB
        assert result["source_type"] == "issue"
        assert result["external_id"] == "123456789"
        assert result["title"] == "Bug: Application crashes on startup"
        assert result["author"] == "testuser"
        assert "bug" in result["metadata"]["labels"]
        assert "critical" in result["metadata"]["labels"]
        assert result["url"] == "https://github.com/test/repo/issues/42"

    def test_transform_comment(self, sample_github_comment: dict[str, Any]):
        adapter = GitHubAdapter()
        result = adapter.transform(sample_github_comment)

        assert result["source"] == SourceType.GITHUB
        assert result["source_type"] == "comment"
        assert result["external_id"] == "987654321"
        assert result["author"] == "contributor1"
        assert "I can reproduce this" in result["content"]

    def test_transform_batch(self, sample_github_issue: dict[str, Any], sample_github_comment: dict[str, Any]):
        adapter = GitHubAdapter()
        batch = adapter.transform_batch([sample_github_issue, sample_github_comment])
        assert len(batch) == 2
        assert batch[0]["source_type"] == "issue"
        assert batch[1]["source_type"] == "comment"

    def test_golden_dataset(self, github_adapter: GitHubAdapter):
        """Golden test: GitHub issue input -> expected normalized output."""
        input_data = _load_golden("github_issue.json")
        result = github_adapter.transform(input_data)

        assert result["source"] == SourceType.GITHUB
        assert result["source_type"] == "issue"
        assert result["external_id"] == "123456789"
        assert result["title"] == "Bug: Application crashes on startup"
        assert "bug" in result["metadata"]["labels"]
        assert result["checksum"] != ""

    def test_properties(self, github_adapter: GitHubAdapter):
        assert github_adapter.source == SourceType.GITHUB
        assert github_adapter.version == "1.0.0"

    def test_empty_body(self):
        adapter = GitHubAdapter()
        issue = {"id": 1, "title": "Title only", "body": None, "state": "open"}
        result = adapter.transform(issue)
        assert result["content"] == ""


class TestHackerNewsAdapter:
    def test_transform_story(self, sample_hn_story: dict[str, Any]):
        adapter = HackerNewsAdapter()
        result = adapter.transform(sample_hn_story)

        assert result["source"] == SourceType.HACKERNEWS
        assert result["source_type"] == "show_hn"
        assert result["external_id"] == "40000001"
        assert result["title"] == "Show HN: I built a tool for real-time code collaboration"
        assert result["author"] == "hnuser1"
        assert result["metadata"]["score"] == 150

    def test_transform_comment(self, sample_hn_comment: dict[str, Any]):
        adapter = HackerNewsAdapter()
        result = adapter.transform(sample_hn_comment)

        assert result["source"] == SourceType.HACKERNEWS
        assert result["source_type"] == "comment"
        assert result["external_id"] == "40000002"
        assert result["author"] == "commenter1"

    def test_transform_askhn(self, sample_hn_askhn: dict[str, Any]):
        adapter = HackerNewsAdapter()
        result = adapter.transform(sample_hn_askhn)

        assert result["source_type"] == "ask_hn"
        assert "Ask HN" in result["title"]

    def test_golden_dataset(self, hn_adapter: HackerNewsAdapter):
        """Golden test: HN story input -> expected normalized output."""
        input_data = _load_golden("hn_story.json")
        result = hn_adapter.transform(input_data)

        assert result["source"] == SourceType.HACKERNEWS
        assert result["source_type"] == "show_hn"
        assert result["external_id"] == "40000001"
        assert result["title"] == "Show HN: I built a tool for real-time code collaboration"
        assert result["metadata"]["score"] == 150
        assert result["checksum"] != ""

    def test_properties(self, hn_adapter: HackerNewsAdapter):
        assert hn_adapter.source == SourceType.HACKERNEWS
        assert hn_adapter.version == "1.0.0"

    def test_empty_items_filtered(self, hn_adapter: HackerNewsAdapter):
        """transform_batch should filter out empty/invalid dicts."""
        result = hn_adapter.transform_batch([{}, {"id": 1, "type": "story", "title": "T"}])
        # Empty dict is filtered out, only the valid story remains
        assert len(result) == 1
        assert result[0]["external_id"] == "1"


class TestProductHuntAdapter:
    def test_transform_post(self, sample_ph_post: dict[str, Any]):
        adapter = ProductHuntAdapter()
        result = adapter.transform(sample_ph_post)

        assert result["source"] == SourceType.PRODUCTHUNT
        assert result["source_type"] == "post"
        assert result["external_id"] == "PH-12345"
        assert result["title"] == "AI Code Review Bot"
        assert result["author"] == "janedev"
        assert result["metadata"]["votes_count"] == 342
        assert result["metadata"]["comments_count"] == 67
        assert "Developer Tools" in result["metadata"]["topics"]
        assert "Artificial Intelligence" in result["metadata"]["topics"]

    def test_transform_comment(self, sample_ph_comment: dict[str, Any]):
        adapter = ProductHuntAdapter()
        result = adapter.transform(sample_ph_comment)

        assert result["source"] == SourceType.PRODUCTHUNT
        assert result["source_type"] == "comment"
        assert result["external_id"] == "PH-C-001"
        assert result["author"] == "Bob Reviewer"
        assert "amazing" in result["content"]
        assert result["metadata"]["post_id"] == "PH-12345"

    def test_transform_batch(self, sample_ph_post: dict[str, Any], sample_ph_comment: dict[str, Any]):
        adapter = ProductHuntAdapter()
        batch = adapter.transform_batch([sample_ph_post, sample_ph_comment])
        assert len(batch) == 2
        assert batch[0]["source_type"] == "post"
        assert batch[1]["source_type"] == "comment"

    def test_golden_dataset(self, ph_adapter: ProductHuntAdapter):
        """Golden test: Product Hunt post input -> expected normalized output."""
        input_data = _load_golden("producthunt_post.json")
        result = ph_adapter.transform(input_data)

        assert result["source"] == SourceType.PRODUCTHUNT
        assert result["source_type"] == "post"
        assert result["external_id"] == "PH-12345"
        assert result["title"] == "AI Code Review Bot"
        assert result["metadata"]["votes_count"] == 342
        assert "Developer Tools" in result["metadata"]["topics"]
        assert result["checksum"] != ""

    def test_properties(self, ph_adapter: ProductHuntAdapter):
        assert ph_adapter.source == SourceType.PRODUCTHUNT
        assert ph_adapter.version == "1.0.0"

    def test_empty_description(self):
        adapter = ProductHuntAdapter()
        post = {"id": "PH-999", "name": "Minimal Post", "tagline": "Just a tagline", "description": "",
                "votesCount": 10, "commentsCount": 0, "url": "", "createdAt": "2026-01-01T00:00:00Z",
                "topics": {"edges": []}, "makers": {"edges": []}}
        result = adapter.transform(post)
        assert result["content"] == "Just a tagline"

    def test_empty_items_filtered(self, ph_adapter: ProductHuntAdapter):
        """transform_batch should filter out empty dicts."""
        result = ph_adapter.transform_batch([{}, {"id": "PH-1", "name": "Test"}])
        assert len(result) == 1
        assert result[0]["external_id"] == "PH-1"


class TestProductHuntParser:
    def test_parse_posts_page(self):
        response_data = {
            "data": {
                "posts": {
                    "edges": [
                        {"node": {"id": "PH-1", "name": "Post 1"}},
                        {"node": {"id": "PH-2", "name": "Post 2"}},
                    ],
                    "pageInfo": {"hasNextPage": True, "endCursor": "cursor-abc"},
                }
            }
        }
        posts, cursor = ProductHuntParser.parse_posts_page(response_data)
        assert len(posts) == 2
        assert cursor == "cursor-abc"

    def test_parse_posts_page_no_next(self):
        response_data = {
            "data": {
                "posts": {
                    "edges": [{"node": {"id": "PH-1", "name": "Post 1"}}],
                    "pageInfo": {"hasNextPage": False, "endCursor": None},
                }
            }
        }
        posts, cursor = ProductHuntParser.parse_posts_page(response_data)
        assert len(posts) == 1
        assert cursor is None

    def test_parse_comments_page(self):
        response_data = {
            "data": {
                "post": {
                    "comments": {
                        "edges": [
                            {"node": {"id": "C-1", "commentBody": "Great!"}},
                        ],
                        "pageInfo": {"hasNextPage": False, "endCursor": None},
                    }
                }
            }
        }
        comments, cursor = ProductHuntParser.parse_comments_page(response_data)
        assert len(comments) == 1
        assert comments[0]["id"] == "C-1"
        assert cursor is None

    def test_extract_topics(self):
        post = {
            "topics": {
                "edges": [
                    {"node": {"name": "AI"}},
                    {"node": {"name": "SaaS"}},
                ]
            }
        }
        topics = ProductHuntParser.extract_topics(post)
        assert topics == ["AI", "SaaS"]

    def test_extract_makers(self):
        post = {
            "makers": {
                "edges": [
                    {"node": {"id": "u1", "name": "Alice", "username": "alice"}},
                ]
            }
        }
        makers = ProductHuntParser.extract_makers(post)
        assert len(makers) == 1
        assert makers[0]["username"] == "alice"


class TestYouTubeAdapter:
    def test_transform_video(self, sample_yt_video: dict[str, Any]):
        adapter = YouTubeAdapter()
        result = adapter.transform(sample_yt_video)

        assert result["source"] == SourceType.YOUTUBE
        assert result["source_type"] == "video"
        assert result["external_id"] == "dQw4w9WgXcQ"
        assert result["title"] == "Building a Startup in 2026: Complete Guide"
        assert result["author"] == "Tech Startup Hub"
        assert result["metadata"]["view_count"] == 125000
        assert result["metadata"]["like_count"] == 4500
        assert result["metadata"]["comment_count"] == 342
        assert result["metadata"]["duration"] == "PT25M30S"
        assert "startup" in result["metadata"]["tags"]

    def test_transform_comment_thread(self, sample_yt_comment_thread: dict[str, Any]):
        adapter = YouTubeAdapter()
        result = adapter.transform(sample_yt_comment_thread)

        assert result["source"] == SourceType.YOUTUBE
        assert result["source_type"] == "comment"
        assert result["external_id"] == "yt-comment-001"
        assert result["author"] == "TechViewer42"
        assert "Great video" in result["content"]
        assert result["metadata"]["like_count"] == 15

    def test_transform_batch(self, sample_yt_video: dict[str, Any], sample_yt_comment_thread: dict[str, Any]):
        adapter = YouTubeAdapter()
        batch = adapter.transform_batch([sample_yt_video, sample_yt_comment_thread])
        assert len(batch) == 2
        assert batch[0]["source_type"] == "video"
        assert batch[1]["source_type"] == "comment"

    def test_golden_dataset(self, yt_adapter: YouTubeAdapter):
        """Golden test: YouTube video input -> expected normalized output."""
        input_data = _load_golden("youtube_video.json")
        result = yt_adapter.transform(input_data)

        assert result["source"] == SourceType.YOUTUBE
        assert result["source_type"] == "video"
        assert result["external_id"] == "dQw4w9WgXcQ"
        assert result["title"] == "Building a Startup in 2026: Complete Guide"
        assert result["metadata"]["view_count"] == 125000
        assert result["checksum"] != ""

    def test_properties(self, yt_adapter: YouTubeAdapter):
        assert yt_adapter.source == SourceType.YOUTUBE
        assert yt_adapter.version == "1.0.0"

    def test_empty_items_filtered(self, yt_adapter: YouTubeAdapter):
        """transform_batch should filter out empty dicts."""
        result = yt_adapter.transform_batch([{}, {"id": "vid1", "kind": "youtube#video", "snippet": {"title": "T"}}])
        assert len(result) == 1
        assert result[0]["external_id"] == "vid1"


class TestYouTubeParser:
    def test_parse_search_results(self):
        response_data = {
            "items": [
                {"id": {"kind": "youtube#video", "videoId": "vid1"}},
                {"id": {"kind": "youtube#video", "videoId": "vid2"}},
            ],
            "nextPageToken": "next-page-token",
        }
        items, token = YouTubeParser.parse_search_results(response_data)
        assert len(items) == 2
        assert token == "next-page-token"

    def test_parse_search_results_no_next(self):
        response_data = {
            "items": [{"id": {"kind": "youtube#video", "videoId": "vid1"}}],
        }
        items, token = YouTubeParser.parse_search_results(response_data)
        assert len(items) == 1
        assert token is None

    def test_parse_videos_response(self):
        response_data = {
            "items": [
                {"id": "vid1", "snippet": {"title": "Video 1"}},
                {"id": "vid2", "snippet": {"title": "Video 2"}},
            ]
        }
        items = YouTubeParser.parse_videos_response(response_data)
        assert len(items) == 2

    def test_parse_comment_threads(self):
        response_data = {
            "items": [
                {"id": "c1", "snippet": {"topLevelComment": {"id": "tc1"}}},
            ],
            "nextPageToken": "comment-cursor",
        }
        items, token = YouTubeParser.parse_comment_threads(response_data)
        assert len(items) == 1
        assert token == "comment-cursor"

    def test_parse_channel_response(self):
        response_data = {
            "items": [{"id": "UC123", "snippet": {"title": "My Channel"}}]
        }
        item = YouTubeParser.parse_channel_response(response_data)
        assert item is not None
        assert item["id"] == "UC123"

    def test_parse_channel_response_empty(self):
        item = YouTubeParser.parse_channel_response({"items": []})
        assert item is None


class TestPlayStoreAdapter:
    def test_transform_review(self, sample_ps_review: dict[str, Any]):
        adapter = PlayStoreAdapter()
        result = adapter.transform(sample_ps_review)

        assert result["source"] == SourceType.PLAYSTORE
        assert result["source_type"] == "review"
        assert result["external_id"] == "abc123-def456-ghi789"
        assert result["author"] == "App Reviewer 42"
        assert result["metadata"]["star_rating"] == 4
        assert "rating:4" in result["tags"]
        assert result["metadata"]["developer_reply"] is None

    def test_transform_review_with_reply(self, sample_ps_review_with_reply: dict[str, Any]):
        adapter = PlayStoreAdapter()
        result = adapter.transform(sample_ps_review_with_reply)

        assert result["source"] == SourceType.PLAYSTORE
        assert result["source_type"] == "review"
        assert result["external_id"] == "xyz789-abc123-def456"
        assert result["metadata"]["star_rating"] == 2
        assert "rating:2" in result["tags"]
        assert result["metadata"]["developer_reply"] is not None
        assert "fixed" in result["metadata"]["developer_reply"]

    def test_transform_batch(self, sample_ps_review: dict[str, Any], sample_ps_review_with_reply: dict[str, Any]):
        adapter = PlayStoreAdapter()
        batch = adapter.transform_batch([sample_ps_review, sample_ps_review_with_reply])
        assert len(batch) == 2
        assert batch[0]["source_type"] == "review"
        assert batch[1]["source_type"] == "review"

    def test_golden_dataset(self, ps_adapter: PlayStoreAdapter):
        """Golden test: Play Store review input -> expected normalized output."""
        input_data = _load_golden("playstore_review.json")
        result = ps_adapter.transform(input_data)

        assert result["source"] == SourceType.PLAYSTORE
        assert result["source_type"] == "review"
        assert result["external_id"] == "abc123-def456-ghi789"
        assert result["author"] == "Tech Enthusiast"
        assert result["metadata"]["star_rating"] == 4
        assert result["checksum"] != ""

    def test_properties(self, ps_adapter: PlayStoreAdapter):
        assert ps_adapter.source == SourceType.PLAYSTORE
        assert ps_adapter.version == "1.0.0"

    def test_empty_items_filtered(self, ps_adapter: PlayStoreAdapter):
        """transform_batch should filter out invalid dicts."""
        result = ps_adapter.transform_batch([{}, {"reviewId": "r1", "userName": "User", "score": 5, "content": "Good"}])
        assert len(result) == 1
        assert result[0]["external_id"] == "r1"

    def test_star_rating_tags(self):
        adapter = PlayStoreAdapter()
        review_1star = {"reviewId": "r1", "userName": "U", "score": 1, "content": "Bad"}
        review_3star = {"reviewId": "r2", "userName": "U", "score": 3, "content": "OK"}
        review_5star = {"reviewId": "r3", "userName": "U", "score": 5, "content": "Great"}

        r1 = adapter.transform(review_1star)
        r2 = adapter.transform(review_3star)
        r3 = adapter.transform(review_5star)

        assert "rating:1" in r1["tags"]
        assert "rating:3" in r2["tags"]
        assert "rating:5" in r3["tags"]

    def test_transform_app_info(self, ps_adapter: PlayStoreAdapter):
        """Test transforming app metadata."""
        app_data = {
            "appId": "com.openai.chatgpt",
            "title": "ChatGPT",
            "developer": "OpenAI",
            "genre": "Productivity",
            "score": 4.7,
            "ratings": 1000000,
            "reviews": 500000,
            "version": "1.2026.300",
        }
        result = ps_adapter.transform(app_data)

        assert result["source"] == SourceType.PLAYSTORE
        assert result["source_type"] == "app_info"
        assert result["external_id"] == "com.openai.chatgpt"
        assert result["title"] == "ChatGPT"
        assert result["metadata"]["score"] == 4.7
        assert "type:app_info" in result["tags"]


class TestPlayStoreParser:
    def test_parse_reviews_page(self):
        reviews = [
            {"reviewId": "r1", "score": 5, "userName": "User1", "content": "Good"},
            {"reviewId": "r2", "score": 3, "userName": "User2", "content": "OK"},
        ]
        result_reviews, token = PlayStoreParser.parse_reviews_page(reviews, "next-page-token")
        assert len(result_reviews) == 2
        assert token == "next-page-token"

    def test_parse_reviews_page_no_next(self):
        reviews = [{"reviewId": "r1", "score": 5, "userName": "User1", "content": "Good"}]
        result_reviews, token = PlayStoreParser.parse_reviews_page(reviews, None)
        assert len(result_reviews) == 1
        assert token is None

    def test_parse_reviews_page_empty_token(self):
        reviews = []
        result_reviews, token = PlayStoreParser.parse_reviews_page(reviews, "")
        assert len(result_reviews) == 0
        assert token is None

    def test_extract_reviewer_name(self):
        review = {"userName": "Test User"}
        name = PlayStoreParser.extract_reviewer_name(review)
        assert name == "Test User"

    def test_extract_review_text(self):
        review = {"content": "Great app!"}
        text = PlayStoreParser.extract_review_text(review)
        assert text == "Great app!"

    def test_extract_review_text_empty(self):
        review = {"content": ""}
        text = PlayStoreParser.extract_review_text(review)
        assert text == ""

    def test_parse_app_info(self):
        app_data = {
            "appId": "com.openai.chatgpt",
            "title": "ChatGPT",
            "developer": "OpenAI",
            "genre": "Productivity",
            "score": 4.7,
            "ratings": 1000000,
            "reviews": 500000,
            "version": "1.2026.300",
            "installs": "100,000,000+",
            "minInstalls": 100000000,
        }
        result = PlayStoreParser.parse_app_info(app_data)
        assert result["title"] == "ChatGPT"
        assert result["package_name"] == "com.openai.chatgpt"
        assert result["category"] == "Productivity"
        assert result["score"] == 4.7

    def test_parse_app_info_empty(self):
        result = PlayStoreParser.parse_app_info({})
        assert result == {}
