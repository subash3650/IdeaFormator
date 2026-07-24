"""Tests for the loader registry."""

from __future__ import annotations

import polars as pl
import pytest

from pain_intelligence.loaders.registry import (
    get_loader_for_file,
    get_all_loaders,
)
from pain_intelligence.loaders.amazon_loader import AmazonLoader
from pain_intelligence.loaders.yelp_loader import YelpLoader
from pain_intelligence.loaders.twitter_loader import TwitterLoader
from pain_intelligence.loaders.reddit_loader import RedditLoader


class TestLoaderRegistry:
    """Tests for auto-detection and registry."""

    def test_all_loaders_registered(self):
        loaders = get_all_loaders()
        loader_names = [cls.__name__ for cls in loaders]
        assert "AmazonLoader" in loader_names
        assert "YelpLoader" in loader_names
        assert "TwitterLoader" in loader_names
        assert "RedditLoader" in loader_names

    def test_detect_amazon(self, tmp_dir, sample_amazon_df):
        csv_path = tmp_dir / "Amazon_Reviews.csv"
        sample_amazon_df.write_csv(csv_path)
        loader = get_loader_for_file(csv_path)
        assert isinstance(loader, AmazonLoader)

    def test_detect_yelp(self, tmp_dir, sample_yelp_df):
        csv_path = tmp_dir / "yelp.csv"
        sample_yelp_df.write_csv(csv_path)
        loader = get_loader_for_file(csv_path)
        assert isinstance(loader, YelpLoader)

    def test_detect_twitter_schema1(self, tmp_dir, sample_twitter_schema1_df):
        csv_path = tmp_dir / "Twitter_Data.csv"
        sample_twitter_schema1_df.write_csv(csv_path)
        loader = get_loader_for_file(csv_path)
        assert isinstance(loader, TwitterLoader)

    def test_detect_reddit(self, tmp_dir, sample_reddit_df):
        csv_path = tmp_dir / "Reddit_Data.csv"
        sample_reddit_df.write_csv(csv_path)
        loader = get_loader_for_file(csv_path)
        assert isinstance(loader, RedditLoader)

    def test_unknown_file_raises(self, tmp_dir):
        csv_path = tmp_dir / "unknown.csv"
        pl.DataFrame({"col_a": [1], "col_b": [2]}).write_csv(csv_path)
        with pytest.raises(ValueError, match="No loader found"):
            get_loader_for_file(csv_path)
