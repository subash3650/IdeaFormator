"""Collector for Product Hunt via the official GraphQL API v2."""

from __future__ import annotations

import os
from typing import Any, Iterator

import httpx

from pain_intelligence.ingestion.collectors.base import BaseCollector, COLLECTOR_VERSION
from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.collectors.producthunt.adapter import ProductHuntAdapter
from pain_intelligence.ingestion.collectors.producthunt.parser import ProductHuntParser
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.models import SourceType, SyncState
from pain_intelligence.ingestion.registry import register_collector
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()

PH_API_URL = "https://api.producthunt.com/v2/api/graphql"

POSTS_QUERY = """
query GetPosts($first: Int!, $after: String, $postedAfter: DateTime) {
  posts(first: $first, after: $after, postedAfter: $postedAfter) {
    edges {
      node {
        id
        name
        tagline
        description
        url
        website
        votesCount
        commentsCount
        createdAt
        thumbnail {
          url
        }
        topics {
          edges {
            node {
              name
            }
          }
        }
        makers {
          id
          name
          username
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

POST_COMMENTS_QUERY = """
query GetPostComments($postId: ID!, $first: Int!, $after: String) {
  post(id: $postId) {
    comments(first: $first, after: $after) {
      edges {
        node {
          id
          commentBody
          createdAt
          author {
            name
            username
          }
          post {
            id
          }
        }
      }
      pageInfo {
        hasNextPage
        endCursor
      }
    }
  }
}
"""


@register_collector("producthunt")
class ProductHuntCollector(BaseCollector):
    """Collects posts and comments from Product Hunt via the GraphQL API v2.

    Supports:
    - OAuth2 token exchange (developer token -> access token)
    - Cursor-based pagination
    - Incremental sync via postedAfter parameter
    - Comment collection per post
    """

    adapter_class = ProductHuntAdapter

    def __init__(self, config: CollectorConfig, client: HttpClient) -> None:
        super().__init__(config, client)
        self._token: str | None = None
        self._access_token: str | None = None
        self._parser = ProductHuntParser()

    @property
    def source(self) -> SourceType:
        return SourceType.PRODUCTHUNT

    def authenticate(self) -> None:
        """Exchange client credentials for a Product Hunt access token.

        Uses OAuth2 client_credentials grant to get a Bearer token.
        Requires PRODUCTHUNT_API_KEY (client_id) and PRODUCTHUNT_API_SECRET (client_secret) in .env.
        """
        client_id = os.environ.get("PRODUCTHUNT_API_KEY")
        client_secret = os.environ.get("PRODUCTHUNT_API_SECRET")

        if not client_id or not client_secret:
            logger.warning("[producthunt] Missing credentials. Set PRODUCTHUNT_API_KEY and "
                           "PRODUCTHUNT_API_SECRET in your .env file.")
            return

        try:
            token_url = "https://api.producthunt.com/v2/oauth/token"
            response = httpx.post(
                token_url,
                json={
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "client_credentials",
                },
                timeout=10,
            )

            if response.status_code == 200:
                data = response.json()
                self._access_token = data.get("access_token")
                logger.debug("[producthunt] Access token obtained via client_credentials grant.")
            else:
                logger.error("[producthunt] Token exchange failed: {} {}", response.status_code, response.text[:200])
        except Exception as e:
            logger.error("[producthunt] Token exchange error: {}", e)

    def health_check(self) -> bool:
        """Verify Product Hunt API is reachable via a simple introspection query."""
        if not self._access_token:
            logger.error("[producthunt] No access token available. Cannot perform health check.")
            return False

        try:
            query = "{ __typename }"
            headers = self._build_headers()
            response = self._client.post(PH_API_URL, json={"query": query}, headers=headers)
            self._api_calls += 1

            if response.status_code == 200:
                logger.info("[producthunt] API healthy.")
                return True
            if response.status_code == 401:
                logger.error("[producthunt] Health check failed: 401 Unauthorized. "
                             "Access token is invalid or expired.")
                return False
            if response.status_code == 403:
                logger.error("[producthunt] Health check failed: 403 Forbidden. "
                             "Access token lacks required permissions.")
                return False
            if response.status_code == 429:
                logger.warning("[producthunt] Health check failed: 429 Rate Limited. "
                               "Try again later.")
                return False

            logger.warning("[producthunt] Health check returned status {}: {}",
                           response.status_code, response.text[:200])
            return False
        except Exception as e:
            logger.error("[producthunt] Health check network error: {}", e)
            return False

    def fetch(self, state: SyncState | None) -> Iterator[list[dict[str, Any]]]:
        """Fetch posts from Product Hunt, yielding batches per page.

        Each batch is a list of post or comment nodes.
        """
        if not self._access_token:
            logger.error("[producthunt] No access token. Skipping fetch.")
            return
        yield from self._fetch_posts(state)

    def _fetch_posts(self, state: SyncState | None) -> Iterator[list[dict[str, Any]]]:
        """Fetch posts with cursor-based pagination."""
        headers = self._build_headers()
        batch_size = min(self._config.batch_size, 50)  # PH max is 50
        max_pages = self._config.max_pages
        cursor: str | None = state.cursor if state and state.cursor else None
        posted_after = state.last_sync.isoformat() if state and state.last_sync else None

        for page in range(max_pages):
            variables: dict[str, Any] = {"first": batch_size, "after": cursor}
            if posted_after:
                variables["postedAfter"] = posted_after

            payload = {"query": POSTS_QUERY, "variables": variables}
            logger.debug("[producthunt] GraphQL query page {}", page + 1)

            response = self._client.post(PH_API_URL, json=payload, headers=headers)
            self._api_calls += 1

            if response.status_code == 401:
                logger.error("[producthunt] Fetch failed: 401 Unauthorized. "
                             "Access token is invalid or expired.")
                break
            if response.status_code == 403:
                logger.error("[producthunt] Fetch failed: 403 Forbidden. "
                             "Access token lacks required permissions.")
                break
            if response.status_code == 429:
                logger.warning("[producthunt] Fetch paused: 429 Rate Limited. "
                               "Try again later.")
                break
            if response.status_code != 200:
                logger.warning(
                    "[producthunt] GraphQL returned {}: {}",
                    response.status_code,
                    response.text[:200],
                )
                break

            data = response.json()
            errors = data.get("errors")
            if errors:
                logger.warning("[producthunt] GraphQL errors: {}", errors)
                break

            posts, end_cursor = self._parser.parse_posts_page(data)
            if not posts:
                break

            # Yield posts
            yield posts

            # Fetch comments for each post
            for post_node in posts:
                post_id = post_node.get("id")
                if post_id:
                    comments = self._fetch_post_comments(post_id, headers)
                    if comments:
                        yield comments

            if not end_cursor:
                break
            cursor = end_cursor

    def _fetch_post_comments(self, post_id: str, headers: dict[str, str]) -> list[dict[str, Any]]:
        """Fetch comments for a single post."""
        all_comments: list[dict[str, Any]] = []
        cursor: str | None = None
        batch_size = min(self._config.batch_size, 50)

        for _ in range(3):  # Limit comment pages per post
            variables: dict[str, Any] = {"postId": post_id, "first": batch_size, "after": cursor}
            payload = {"query": POST_COMMENTS_QUERY, "variables": variables}

            response = self._client.post(PH_API_URL, json=payload, headers=headers)
            self._api_calls += 1

            if response.status_code != 200:
                break

            data = response.json()
            errors = data.get("errors")
            if errors:
                break

            comments, end_cursor = self._parser.parse_comments_page(data)
            all_comments.extend(comments)

            if not end_cursor:
                break
            cursor = end_cursor

        return all_comments

    def _build_headers(self) -> dict[str, str]:
        """Build request headers with Bearer token."""
        headers = {"Content-Type": "application/json"}
        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"
        return headers
