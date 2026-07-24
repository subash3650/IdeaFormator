"""Tests for the pipeline orchestrator."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from pain_intelligence.pipeline.orchestrator import Orchestrator
from pain_intelligence.utils.config import load_config, get_nested
from loguru import logger as _loguru_logger


class TestOrchestrator:
    """Tests for the pipeline Orchestrator."""

    def test_load_default_config(self):
        config = load_config("configs/default.yaml")
        assert "paths" in config
        assert "processing" in config
        assert "preprocessing" in config

    def test_get_nested(self):
        config = {"a": {"b": {"c": 42}}}
        assert get_nested(config, "a", "b", "c") == 42
        assert get_nested(config, "a", "x", default="fallback") == "fallback"
        assert get_nested(config, "x", default=10) == 10

    def test_orchestrator_init(self):
        orch = Orchestrator(config_path="configs/default.yaml")
        assert orch.config is not None

    def test_full_pipeline_with_sample_data(self, tmp_dir, sample_amazon_df):
        """Integration test: run the pipeline on a sample CSV."""
        raw_dir = tmp_dir / "Datasets"
        outputs_dir = tmp_dir / "outputs"
        raw_dir.mkdir()

        sample_amazon_df.write_csv(raw_dir / "Amazon_Reviews.csv")

        config_path = tmp_dir / "test_config.yaml"
        config_path.write_text(
            f"""
paths:
  raw_datasets_dir: "{raw_dir.as_posix()}"
  outputs_dir: "{outputs_dir.as_posix()}"
  removed_dir: "{outputs_dir.as_posix()}"

processing:
  chunk_size: 100
  min_document_length: 5
  supported_languages: []

output:
  formats:
    - "csv"

preprocessing:
  fix_encoding: true
  remove_html: true
  normalize_unicode: true
  normalize_whitespace: true
  remove_urls: true
  process_emojis: true
  lowercase: false
  remove_duplicates: true
  detect_language: false
  max_document_length: 100000

logging:
  level: "WARNING"
  file: null
  rotation: "10 MB"
  retention: "1 day"
""",
            encoding="utf-8",
        )

        orch = Orchestrator(config_path=str(config_path))
        stats = orch.run()

        assert stats["total_processed"] > 0
        assert (outputs_dir / "dataset_statistics.json").exists()
        assert (outputs_dir / "processing_report.json").exists()

        with open(outputs_dir / "dataset_statistics.json", encoding="utf-8") as f:
            ds_stats = json.load(f)
        assert ds_stats["total_documents"] > 0
        assert "platform_distribution" in ds_stats

    def test_full_pipeline_with_all_platforms(self, tmp_dir):
        """Integration test with all 4 platforms."""
        raw_dir = tmp_dir / "Datasets"
        outputs_dir = tmp_dir / "outputs"
        raw_dir.mkdir()

        pl.DataFrame({
            "Reviewer Name": ["Alice"],
            "Profile Link": ["/u/1"],
            "Country": ["US"],
            "Review Count": ["1 review"],
            "Review Date": ["2024-01-15T10:30:00.000Z"],
            "Rating": ["Rated 5 out of 5 stars"],
            "Review Title": ["Amazing"],
            "Review Text": ["This product changed my life for the better."],
            "Date of Experience": ["January 10, 2024"],
        }).write_csv(raw_dir / "Amazon_Reviews.csv")

        pl.DataFrame({
            "business_id": ["biz1"],
            "date": ["2024-01-15"],
            "review_id": ["rev1"],
            "stars": [4],
            "text": ["Good restaurant with nice ambiance."],
            "type": ["review"],
            "user_id": ["u1"],
            "cool": [0],
            "useful": [1],
            "funny": [0],
        }).write_csv(raw_dir / "yelp.csv")

        pl.DataFrame({
            "business_id": [100],
            "Location": ["Gaming"],
            "type": ["Positive"],
            "text": ["Love this game so much, highly recommend it!"],
        }).write_csv(raw_dir / "Twitter_Data.csv")

        pl.DataFrame({
            "clean_comment": ["This subreddit has amazing content and helpful people."],
            "category": ["1"],
        }).write_csv(raw_dir / "Reddit_Data.csv")

        config_path = tmp_dir / "test_config.yaml"
        config_path.write_text(
            f"""
paths:
  raw_datasets_dir: "{raw_dir.as_posix()}"
  outputs_dir: "{outputs_dir.as_posix()}"
  removed_dir: "{outputs_dir.as_posix()}"

processing:
  chunk_size: 100
  min_document_length: 5
  supported_languages: []

output:
  formats:
    - "csv"

preprocessing:
  fix_encoding: true
  remove_html: true
  normalize_unicode: true
  normalize_whitespace: true
  remove_urls: true
  process_emojis: true
  lowercase: false
  remove_duplicates: true
  detect_language: false
  max_document_length: 100000

logging:
  level: "WARNING"
  file: null
  rotation: "10 MB"
  retention: "1 day"
""",
            encoding="utf-8",
        )

        orch = Orchestrator(config_path=str(config_path))
        stats = orch.run()

        assert stats["total_processed"] >= 4
        platforms_seen = {ds["platform"] for ds in stats["datasets_processed"]}
        assert "amazon" in platforms_seen
        assert "yelp" in platforms_seen
        assert "twitter" in platforms_seen
        assert "reddit" in platforms_seen
