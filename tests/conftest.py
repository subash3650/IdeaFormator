"""Shared test fixtures."""

from __future__ import annotations

import tempfile
from pathlib import Path

import polars as pl
import pytest
from loguru import logger as _loguru_logger


@pytest.fixture(autouse=True)
def _cleanup_loguru():
    """Remove all loguru handlers after each test to prevent file locks."""
    yield
    _loguru_logger.remove()


@pytest.fixture
def sample_amazon_df() -> pl.DataFrame:
    """Sample Amazon review DataFrame."""
    return pl.DataFrame({
        "Reviewer Name": ["Alice", "Bob"],
        "Profile Link": ["/users/1", "/users/2"],
        "Country": ["US", "GB"],
        "Review Count": ["5 reviews", "10 reviews"],
        "Review Date": ["2024-01-15T10:30:00.000Z", "2024-01-16T11:00:00.000Z"],
        "Rating": ["Rated 4 out of 5 stars", "Rated 2 out of 5 stars"],
        "Review Title": ["Great product", "Terrible experience"],
        "Review Text": [
            "This product is amazing and works perfectly.",
            "Worst purchase I ever made. Complete waste of money.",
        ],
        "Date of Experience": ["January 10, 2024", "January 12, 2024"],
    })


@pytest.fixture
def sample_yelp_df() -> pl.DataFrame:
    """Sample Yelp review DataFrame."""
    return pl.DataFrame({
        "business_id": ["biz1", "biz2"],
        "date": ["2024-01-15", "2024-01-16"],
        "review_id": ["rev1", "rev2"],
        "stars": [5, 1],
        "text": [
            "Excellent food and great service!",
            "Disgusting food and rude staff.",
        ],
        "type": ["review", "review"],
        "user_id": ["user1", "user2"],
        "cool": [2, 0],
        "useful": [5, 1],
        "funny": [0, 3],
    })


@pytest.fixture
def sample_twitter_schema1_df() -> pl.DataFrame:
    """Sample Twitter DataFrame (schema 1)."""
    return pl.DataFrame({
        "business_id": [2401, 2401],
        "Location": ["Borderlands", "Borderlands"],
        "type": ["Positive", "Negative"],
        "text": ["Great game, loving it!", "This game is terrible."],
    })


@pytest.fixture
def sample_twitter_schema2_df() -> pl.DataFrame:
    """Sample Twitter DataFrame (schema 2)."""
    return pl.DataFrame({
        "clean_text": ["Great product", "Terrible service"],
        "category": ["1", "-1"],
    })


@pytest.fixture
def sample_reddit_df() -> pl.DataFrame:
    """Sample Reddit DataFrame."""
    return pl.DataFrame({
        "clean_comment": [
            "This community is wonderful and supportive!",
            "I had a terrible experience with this product.",
        ],
        "category": ["1", "-1"],
    })


@pytest.fixture
def tmp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)
