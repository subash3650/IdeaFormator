"""API collectors — importing triggers @register_collector decorators."""

from pain_intelligence.ingestion.collectors.github import GitHubCollector
from pain_intelligence.ingestion.collectors.hackernews import HackerNewsCollector
from pain_intelligence.ingestion.collectors.producthunt import ProductHuntCollector
from pain_intelligence.ingestion.collectors.youtube import YouTubeCollector
from pain_intelligence.ingestion.collectors.playstore import PlayStoreCollector

__all__ = [
    "GitHubCollector",
    "HackerNewsCollector",
    "ProductHuntCollector",
    "YouTubeCollector",
    "PlayStoreCollector",
]
