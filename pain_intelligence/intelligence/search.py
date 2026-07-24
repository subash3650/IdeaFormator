"""Simple keyword document search."""

from __future__ import annotations

import polars as pl


class DocumentSearch:
    """Simple keyword search over the processed dataset.
    
    Supports: basic keyword, platform/country/rating filters.
    Advanced: regex, boolean search deferred to Phase 2.
    """

    def __init__(self, data_path: str = "outputs/processed.parquet") -> None:
        self.data_path = data_path
        self._df: pl.DataFrame | None = None

    def _load(self) -> pl.DataFrame:
        if self._df is None:
            self._df = pl.read_parquet(self.data_path)
        return self._df

    def search(
        self,
        query: str = "",
        platform: str | None = None,
        country: str | None = None,
        rating_min: float | None = None,
        rating_max: float | None = None,
        text_col: str = "text",
        limit: int = 100,
    ) -> pl.DataFrame:
        df = self._load()

        if query:
            df = df.filter(pl.col(text_col).str.contains(query, literal=True))

        if platform:
            df = df.filter(pl.col("platform") == platform)

        if country:
            df = df.filter(pl.col("country") == country)

        if rating_min is not None:
            df = df.filter(pl.col("rating") >= rating_min)
        if rating_max is not None:
            df = df.filter(pl.col("rating") <= rating_max)

        return df.head(limit)

    def get_platforms(self) -> list[str]:
        df = self._load()
        return df["platform"].unique().sort().to_list()

    def get_countries(self) -> list[str]:
        df = self._load()
        return df["country"].drop_nulls().unique().sort().to_list()

    def get_rating_range(self) -> tuple[float, float]:
        df = self._load()
        r = df["rating"].drop_nulls()
        if r.len() == 0:
            return (0.0, 0.0)
        return (float(r.min()), float(r.max()))