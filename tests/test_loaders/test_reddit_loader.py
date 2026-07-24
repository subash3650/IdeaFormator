"""Tests for the Reddit loader."""

from __future__ import annotations

import polars as pl
import pytest

from pain_intelligence.loaders.reddit_loader import RedditLoader
from pain_intelligence.schema.document import Platform


class TestRedditLoader:
    """Tests for RedditLoader."""

    @pytest.fixture
    def loader(self):
        return RedditLoader()

    def test_detect_reddit_columns(self, loader):
        cols = ["clean_comment", "category"]
        assert loader._detect(cols) is True

    def test_detect_wrong_columns(self, loader):
        cols = ["text", "label"]
        assert loader._detect(cols) is False

    def test_transform_row(self, loader, sample_reddit_df):
        row = sample_reddit_df.row(0, named=True)
        doc = loader._transform_row(row)

        assert doc.platform == Platform.REDDIT
        assert doc.source_dataset == "Reddit_Data.csv"
        assert "wonderful" in doc.text.lower()
        assert doc.metadata["sentiment_category"] == "1"

    def test_transform_chunk(self, loader, sample_reddit_df):
        docs = loader.transform_chunk(sample_reddit_df)
        assert len(docs) == 2
        assert all(d.platform == Platform.REDDIT for d in docs)

    def test_load_file(self, loader, tmp_dir, sample_reddit_df):
        csv_path = tmp_dir / "reddit_test.csv"
        sample_reddit_df.write_csv(csv_path)

        chunks = list(loader.load(csv_path, chunk_size=100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2
