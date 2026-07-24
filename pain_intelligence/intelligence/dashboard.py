"""Minimal 3-page Streamlit explorer for the Knowledge Extraction Engine."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import streamlit as st
except ImportError:
    print("Streamlit not installed. Install with: pip install streamlit")
    sys.exit(1)

import polars as pl

from pain_intelligence.intelligence.search import DocumentSearch
from pain_intelligence.knowledge.store import KnowledgeStore

STORE_DIR = Path("pain_intelligence/knowledge")
DATA_PATH = "outputs/processed.parquet"


def main() -> None:
    st.set_page_config(page_title="Pain Intelligence Explorer", layout="wide")
    st.sidebar.title("Pain Intelligence")
    page = st.sidebar.radio("Pages", ["Overview", "Search", "Problem Signals"])

    store = KnowledgeStore(STORE_DIR)
    search_engine = DocumentSearch(DATA_PATH)

    if page == "Overview":
        _show_overview(store)
    elif page == "Search":
        _show_search(search_engine)
    elif page == "Problem Signals":
        _show_signals(store)


def _show_overview(store: KnowledgeStore) -> None:
    st.header("Overview")

    df = pl.read_parquet(DATA_PATH)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Documents", f"{df.height:,}")
    col2.metric("Platforms", df["platform"].n_unique())
    col3.metric("Avg Length", f"{df['document_length'].mean():.0f}")
    col4.metric("Has Rating", f"{df['rating'].drop_nulls().count():,}")

    st.subheader("Platform Distribution")
    plat = df["platform"].value_counts().to_dict()
    if plat:
        st.bar_chart(plat, x="platform", y="count")

    st.subheader("Rating Distribution (top 20)")
    ratings = df["rating"].drop_nulls()
    if ratings.len() > 0:
        rating_counts = ratings.value_counts().sort("rating").to_dict()
        st.bar_chart(rating_counts, x="rating", y="count")

    st.subheader("Document Quality")
    quality = pl.DataFrame({
        "Metric": ["Total", "With Rating", "With Country", "With Title", "With Language"],
        "Count": [
            df.height,
            df["rating"].drop_nulls().count(),
            df["country"].drop_nulls().count(),
            df["title"].drop_nulls().count(),
            df["language"].drop_nulls().count(),
        ],
    })
    st.dataframe(quality, use_container_width=True)

    # Knowledge assets
    st.subheader("Knowledge Assets")
    for name in ["observations", "evidence", "problem_signals"]:
        if store.asset_exists(name):
            asset = store.read_asset(name)
            st.write(f"**{name}**: {asset.height:,} records, {len(asset.columns)} columns")


def _show_search(search_engine: DocumentSearch) -> None:
    st.header("Document Search")

    query = st.text_input("Search query", value="")
    col1, col2 = st.columns(2)
    platforms = [""] + search_engine.get_platforms()
    platform = col1.selectbox("Platform", platforms)
    countries = [""] + search_engine.get_countries()
    country = col2.selectbox("Country", countries)
    limit = st.slider("Max results", 10, 500, 100)

    if query:
        with st.spinner("Searching..."):
            results = search_engine.search(
                query=query,
                platform=platform or None,
                country=country or None,
                limit=limit,
            )
        st.write(f"Found {len(results)} documents")
        for row in results.iter_rows(named=True):
            text = (row.get("text") or "")[:300]
            rating = row.get("rating")
            platform_val = row.get("platform", "")
            country_val = row.get("country", "")
            with st.expander(f"[{platform_val}] {'★' * int(rating or 0)} - {text[:80]}..."):
                st.write(f"**Rating**: {rating}")
                st.write(f"**Platform**: {platform_val}")
                st.write(f"**Country**: {country_val}")
                st.write(f"**Text**: {text}")
    else:
        st.info("Enter a query to search documents.")


def _show_signals(store: KnowledgeStore) -> None:
    st.header("Problem Signals")

    if not store.asset_exists("problem_signals"):
        st.warning("No problem signals found. Run the intelligence pipeline first.")
        return

    signals = store.read_asset("problem_signals")
    st.write(f"Total signals: {len(signals)}")

    sort_col = st.selectbox("Sort by", ["confidence", "document_count", "avg_rating"])
    ascending = st.checkbox("Ascending", False)
    sorted_signals = signals.sort(sort_col, descending=not ascending)

    for row in sorted_signals.iter_rows(named=True):
        signal_key = row.get("signal_key", "?")
        count = row.get("document_count", 0)
        rating = row.get("avg_rating")
        conf = row.get("confidence", 0)
        entity = row.get("entity", "")
        category = row.get("category", "")

        with st.expander(f"[{conf:.2f}] {signal_key[:60]} — {count} docs, {'★' * int(rating or 0)}{'☆' * (5 - int(rating or 0))}"):
            st.write(f"**Category**: {category}")
            st.write(f"**Entity**: {entity}")
            st.write(f"**Documents**: {count}")
            st.write(f"**Avg Rating**: {rating:.2f}")
            st.write(f"**Confidence**: {conf:.4f}")

    if st.button("Export to JSON"):
        import json
        out = signals.to_dicts()
        path = STORE_DIR.parent / "reports" / "signals_export.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        st.success(f"Exported to {path}")


if __name__ == "__main__":
    main()