"""Dataset loaders for various platforms.

Importing this module registers all built-in loaders with the registry.
"""

from pain_intelligence.loaders.amazon_loader import AmazonLoader  # noqa: F401
from pain_intelligence.loaders.yelp_loader import YelpLoader  # noqa: F401
from pain_intelligence.loaders.twitter_loader import TwitterLoader  # noqa: F401
from pain_intelligence.loaders.reddit_loader import RedditLoader  # noqa: F401
