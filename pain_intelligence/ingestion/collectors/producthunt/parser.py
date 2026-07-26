"""Parser for Product Hunt GraphQL API responses."""

from __future__ import annotations

from typing import Any


class ProductHuntParser:
    """Extracts structured data from Product Hunt GraphQL response payloads."""

    @staticmethod
    def parse_posts_page(response_data: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        """Parse a posts listing GraphQL response.

        Returns (posts, endCursor) tuple.
        """
        data = response_data.get("data", {})
        posts_connection = data.get("posts", {})
        edges = posts_connection.get("edges", [])
        page_info = posts_connection.get("pageInfo", {})

        posts: list[dict[str, Any]] = []
        for edge in edges:
            node = edge.get("node", {})
            if node:
                posts.append(node)

        end_cursor = page_info.get("endCursor") if page_info.get("hasNextPage") else None
        return posts, end_cursor

    @staticmethod
    def parse_comments_page(response_data: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
        """Parse a post's comments GraphQL response.

        Returns (comments, endCursor) tuple.
        """
        data = response_data.get("data", {})
        post = data.get("post", {})
        comments_connection = post.get("comments", {})
        edges = comments_connection.get("edges", [])
        page_info = comments_connection.get("pageInfo", {})

        comments: list[dict[str, Any]] = []
        for edge in edges:
            node = edge.get("node", {})
            if node:
                comments.append(node)

        end_cursor = page_info.get("endCursor") if page_info.get("hasNextPage") else None
        return comments, end_cursor

    @staticmethod
    def extract_topics(post_node: dict[str, Any]) -> list[str]:
        """Extract topic names from a post node."""
        topics_connection = post_node.get("topics", {})
        edges = topics_connection.get("edges", [])
        return [edge.get("node", {}).get("name", "") for edge in edges if edge.get("node", {}).get("name")]

    @staticmethod
    def extract_makers(post_node: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract maker info from a post node."""
        makers_data = post_node.get("makers", [])

        # Handle both formats: list of dicts (new) or connection with edges (old)
        if isinstance(makers_data, list):
            makers: list[dict[str, Any]] = []
            for maker in makers_data:
                if isinstance(maker, dict):
                    makers.append({
                        "id": maker.get("id", ""),
                        "name": maker.get("name", ""),
                        "username": maker.get("username", ""),
                    })
            return makers

        # Legacy connection format
        edges = makers_data.get("edges", [])
        makers = []
        for edge in edges:
            node = edge.get("node", {})
            if node:
                makers.append({
                    "id": node.get("id", ""),
                    "name": node.get("name", ""),
                    "username": node.get("username", ""),
                })
        return makers
