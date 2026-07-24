"""Tests for the Yelp loader."""

from __future__ import annotations

import polars as pl
import pytest

from pain_intelligence.loaders.yelp_loader import YelpLoader
from pain_intelligence.schema.document import Platform


class TestYelpLoader:
    """Tests for YelpLoader."""

    @pytest.fixture
    def loader(self):
        return YelpLoader()

    def test_detect_yelp_columns(self, loader):
        cols = ["business_id", "date", "review_id", "stars", "text", "type", "user_id"]
        assert loader._detect(cols) is True

    def test_detect_wrong_columns(self, loader):
        cols = ["foo", "bar"]
        assert loader._detect(cols) is False

    def test_transform_row(self, loader, sample_yelp_df):
        row = sample_yelp_df.row(0, named=True)
        doc = loader._transform_row(row)

        assert doc.platform == Platform.YELP
        assert doc.source_dataset == "yelp.csv"
        assert doc.rating == 5.0
        assert "Excellent" in doc.text

    def test_transform_chunk(self, loader, sample_yelp_df):
        docs = loader.transform_chunk(sample_yelp_df)
        assert len(docs) == 2
        assert docs[0].rating == 5.0
        assert docs[1].rating == 1.0

    def test_metadata_contains_business_id(self, loader, sample_yelp_df):
        row = sample_yelp_df.row(0, named=True)
        doc = loader._transform_row(row)
        assert doc.metadata["business_id"] == "biz1"
        assert doc.metadata["cool"] == 2

    def test_load_file(self, loader, tmp_dir, sample_yelp_df):
        csv_path = tmp_dir / "yelp_test.csv"
        sample_yelp_df.write_csv(csv_path)

        chunks = list(loader.load(csv_path, chunk_size=100))
        assert len(chunks) == 1
        assert len(chunks[0]) == 2
