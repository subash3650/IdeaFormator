"""Tests for the Amazon loader."""

from __future__ import annotations

import polars as pl
import pytest

from pain_intelligence.loaders.amazon_loader import AmazonLoader
from pain_intelligence.schema.document import Platform


class TestAmazonLoader:
    """Tests for AmazonLoader."""

    @pytest.fixture
    def loader(self):
        return AmazonLoader()

    def test_detect_amazon_columns(self, loader):
        cols = [
            "Reviewer Name", "Profile Link", "Country", "Review Count",
            "Review Date", "Rating", "Review Title", "Review Text",
            "Date of Experience",
        ]
        assert loader._detect(cols) is True

    def test_detect_wrong_columns(self, loader):
        cols = ["foo", "bar", "baz"]
        assert loader._detect(cols) is False

    def test_detect_partial_columns(self, loader):
        cols = ["Reviewer Name", "Review Text"]  # Missing Rating
        assert loader._detect(cols) is False

    def test_transform_row(self, loader, sample_amazon_df):
        row = sample_amazon_df.row(0, named=True)
        doc = loader._transform_row(row)

        assert doc.platform == Platform.AMAZON
        assert doc.source_dataset == "Amazon_Reviews.csv"
        assert doc.author == "Alice"
        assert doc.country == "US"
        assert doc.title == "Great product"
        assert doc.rating == 4.0
        assert "amazing" in doc.text.lower()

    def test_transform_second_row(self, loader, sample_amazon_df):
        row = sample_amazon_df.row(1, named=True)
        doc = loader._transform_row(row)

        assert doc.rating == 2.0
        assert doc.country == "GB"

    def test_extract_rating_number(self, loader):
        assert loader._extract_rating_number("Rated 3 out of 5 stars") == 3.0
        assert loader._extract_rating_number("Rated 1 out of 5 stars") == 1.0
        assert loader._extract_rating_number("no rating here") is None

    def test_transform_chunk(self, loader, sample_amazon_df):
        docs = loader.transform_chunk(sample_amazon_df)
        assert len(docs) == 2
        assert all(d.platform == Platform.AMAZON for d in docs)

    def test_load_file(self, loader, tmp_dir, sample_amazon_df):
        csv_path = tmp_dir / "amazon_test.csv"
        sample_amazon_df.write_csv(csv_path)

        chunks = list(loader.load(csv_path, chunk_size=100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2
