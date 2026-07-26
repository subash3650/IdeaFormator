"""Adapter for transforming GitHub API responses into normalized dicts."""

from __future__ import annotations

from typing import Any

from pain_intelligence.ingestion.adapters.base import BaseAdapter
from pain_intelligence.ingestion.models import SourceType
from pain_intelligence.ingestion.utils import compute_checksum, compute_document_id, parse_timestamp


class GitHubAdapter(BaseAdapter):
    """Transforms GitHub Issues/Comments API responses into normalized dicts."""

    @property
    def source(self) -> SourceType:
        return SourceType.GITHUB

    @property
    def version(self) -> str:
        return "1.0.0"

    def transform(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        """Transform a single GitHub issue or comment dict.

        GitHub comments have an 'issue_url' field but no 'repository_url'.
        GitHub issues have a 'repository_url' field or a 'number' field.
        """
        if "issue_url" in raw_response and "repository_url" not in raw_response:
            return self._transform_comment(raw_response)
        return self._transform_issue(raw_response)

    def transform_batch(self, raw_responses: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Transform a batch of GitHub API responses."""
        return [self.transform(item) for item in raw_responses]

    def _transform_issue(self, issue: dict[str, Any]) -> dict[str, Any]:
        """Transform a GitHub issue into a normalized dict."""
        external_id = str(issue.get("id", ""))
        number = issue.get("number", "")
        repo_name = issue.get("repository_url", "").split("/")[-1] if issue.get("repository_url") else ""
        state = issue.get("state", "")
        title = issue.get("title", "")
        body = issue.get("body", "") or ""

        # Labels
        labels = [lbl.get("name", "") for lbl in issue.get("labels", []) if lbl.get("name")]
        label_strs = [f"label:{lbl}" for lbl in labels]

        # Build tags from labels + state + type
        tags = label_strs + [f"state:{state}", "type:issue"]

        # Metadata for downstream analysis
        metadata = {
            "number": number,
            "state": state,
            "locked": issue.get("locked", False),
            "comments_count": issue.get("comments", 0),
            "author_association": issue.get("author_association", ""),
            "repository": repo_name,
            "labels": labels,
            "milestone": (issue.get("milestone") or {}).get("title"),
            "assignees": [a.get("login", "") for a in issue.get("assignees", [])],
            "reactions": (issue.get("reactions") or {}).get("total_count", 0),
        }

        document_id = compute_document_id("github", external_id)
        checksum = compute_checksum(f"{title}\n{body}")

        return {
            "document_id": document_id,
            "source": SourceType.GITHUB,
            "source_type": "issue",
            "external_id": external_id,
            "title": title,
            "content": body,
            "author": (issue.get("user") or {}).get("login", ""),
            "created_at": parse_timestamp(issue.get("created_at")),
            "updated_at": parse_timestamp(issue.get("updated_at")),
            "language": None,  # GitHub issues don't have a language field
            "url": issue.get("html_url", ""),
            "tags": tags,
            "categories": [f"repo:{repo_name}"] if repo_name else [],
            "metadata": metadata,
            "raw_json": issue,
            "checksum": checksum,
            "collector_version": self.version,
        }

    def _transform_comment(self, comment: dict[str, Any]) -> dict[str, Any]:
        """Transform a GitHub issue comment into a normalized dict."""
        external_id = str(comment.get("id", ""))
        body = comment.get("body", "") or ""
        issue_url = comment.get("issue_url", "")

        tags = ["type:comment"]

        metadata = {
            "issue_url": issue_url,
            "author_association": comment.get("author_association", ""),
            "reactions": (comment.get("reactions") or {}).get("total_count", 0),
        }

        document_id = compute_document_id("github_comment", external_id)
        checksum = compute_checksum(body)

        return {
            "document_id": document_id,
            "source": SourceType.GITHUB,
            "source_type": "comment",
            "external_id": external_id,
            "title": None,
            "content": body,
            "author": (comment.get("user") or {}).get("login", ""),
            "created_at": parse_timestamp(comment.get("created_at")),
            "updated_at": parse_timestamp(comment.get("updated_at")),
            "language": None,
            "url": comment.get("html_url", ""),
            "tags": tags,
            "categories": [],
            "metadata": metadata,
            "raw_json": comment,
            "checksum": checksum,
            "collector_version": self.version,
        }
