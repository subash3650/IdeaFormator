"""Tests for the Twitter loader."""

from __future__ import annotations

import polars as pl
import pytest

from pain_intelligence.loaders.twitter_loader import TwitterLoader
from pain_intelligence.schema.document import Platform


class TestTwitterLoader:
    """Tests for TwitterLoader."""

    @pytest.fixture
    def loader(self):
        return TwitterLoader()

    def test_detect_schema1(self, loader):
        cols = ["business_id", "Location", "type", "text"]
        assert loader._detect(cols) is True

    def test_detect_schema2(self, loader):
        cols = ["clean_text", "category"]
        assert loader._detect(cols) is True

    def test_detect_wrong(self, loader):
        cols = ["foo", "bar"]
        assert loader._detect(cols) is False

    def test_transform_schema1(self, loader, sample_twitter_schema1_df):
        row = sample_twitter_schema1_df.row(0, named=True)
        doc = loader._transform_row(row)

        assert doc.platform == Platform.TWITTER
        assert doc.source_dataset == "Twitter_Data.csv"
        assert doc.location == "Borderlands"
        assert doc.metadata["sentiment_label"] == "Positive"

    def test_transform_schema2(self, loader, sample_twitter_schema2_df):
        row = sample_twitter_schema2_df.row(0, named=True)
        doc = loader._transform_row(row)

        assert doc.platform == Platform.TWITTER
        assert doc.source_dataset == "twitter_training.csv"
        assert doc.metadata["sentiment_category"] == "1"

    def test_transform_chunk_schema1(self, loader, sample_twitter_schema1_df):
        docs = loader.transform_chunk(sample_twitter_schema1_df)
        assert len(docs) == 2

    def test_transform_chunk_schema2(self, loader, sample_twitter_schema2_df):
        docs = loader.transform_chunk(sample_twitter_schema2_df)
        assert len(docs) == 2
