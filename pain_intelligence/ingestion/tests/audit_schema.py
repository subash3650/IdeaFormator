"""Audit field types in real PlayStore data to find the inconsistent field."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from pain_intelligence.ingestion.collectors.playstore.adapter import PlayStoreAdapter
from pain_intelligence.ingestion.models import RawDocument

# Fetch real reviews
from google_play_scraper import reviews as gp_reviews, Sort

adapter = PlayStoreAdapter()

print("Fetching PlayStore reviews...")
result, token = gp_reviews(
    "com.openai.chatgpt",
    lang="en",
    country="us",
    sort=Sort.NEWEST,
    count=100,
)
print(f"Fetched {len(result)} reviews\n")

# Transform and normalize
transformed = adapter.transform_batch(result)
print(f"Transformed {len(transformed)} records\n")

# Create RawDocuments
docs: list[RawDocument] = []
errors = 0
for t in transformed:
    try:
        doc = RawDocument(**t)
        docs.append(doc)
    except Exception as e:
        errors += 1
        print(f"  Normalization failed: {e}")
print(f"Created {len(docs)} RawDocuments ({errors} errors)\n")

# --- AUDIT 1: RawDocument fields (Pydantic model) ---
print("=" * 60)
print("AUDIT 1: RawDocument model fields (via model_dump)")
print("=" * 60)

field_types: dict[str, Counter] = {}
for doc in docs:
    raw = doc.model_dump()
    for field_name, value in raw.items():
        if field_name not in field_types:
            field_types[field_name] = Counter()
        field_types[field_name][type(value).__name__] += 1

for field_name, counts in sorted(field_types.items()):
    if len(counts) > 1:
        print(f"\n*** INCONSISTENT FIELD: {field_name} ***")
        print(f"    Types: {dict(counts)}")
        # Find first doc with minority type
        majority = counts.most_common(1)[0][0]
        for doc in docs:
            raw = doc.model_dump()
            actual = type(raw[field_name]).__name__
            if actual != majority:
                print(f"    Offending document_id: {doc.document_id}")
                print(f"    Offending source: {doc.source}")
                print(f"    Value: {repr(raw[field_name])[:120]}")
                break
    else:
        tname, tcount = list(counts.items())[0]
        print(f"  {field_name:25s} {tname:15s} (all {tcount})")

# --- AUDIT 2: to_flat_dict() fields ---
print("\n" + "=" * 60)
print("AUDIT 2: to_flat_dict() output")
print("=" * 60)

flat_field_types: dict[str, Counter] = {}
for doc in docs:
    flat = doc.to_flat_dict()
    for field_name, value in flat.items():
        if field_name not in flat_field_types:
            flat_field_types[field_name] = Counter()
        flat_field_types[field_name][type(value).__name__] += 1

for field_name, counts in sorted(flat_field_types.items()):
    if len(counts) > 1:
        print(f"\n*** INCONSISTENT FIELD: {field_name} ***")
        print(f"    Types: {dict(counts)}")
        majority = counts.most_common(1)[0][0]
        for doc in docs:
            flat = doc.to_flat_dict()
            actual = type(flat[field_name]).__name__
            if actual != majority:
                print(f"    Offending document_id: {doc.document_id}")
                print(f"    Value: {repr(flat[field_name])[:120]}")
                break
    else:
        tname, tcount = list(counts.items())[0]
        print(f"  {field_name:25s} {tname:15s} (all {tcount})")

# --- AUDIT 3: Polars DataFrame construction ---
print("\n" + "=" * 60)
print("AUDIT 3: Polars DataFrame from to_flat_dict()")
print("=" * 60)

try:
    rows = [doc.to_flat_dict() for doc in docs]
    df = pl.DataFrame(rows)
    print("  Polars DataFrame creation: SUCCESS")
    print(f"  Shape: {df.shape}")
    for col in df.columns:
        print(f"  {col:25s} {df[col].dtype}")
except Exception as e:
    print(f"  Polars DataFrame creation: FAILED")
    print(f"  Error: {e}")

# --- AUDIT 4: Check metadata contents ---
print("\n" + "=" * 60)
print("AUDIT 4: Raw metadata types inside model_dump")
print("=" * 60)

meta_field_types: dict[str, Counter] = {}
for doc in docs:
    raw = doc.model_dump()
    meta = raw.get("metadata", {})
    if isinstance(meta, dict):
        for k, v in meta.items():
            if k not in meta_field_types:
                meta_field_types[k] = Counter()
            meta_field_types[k][type(v).__name__] += 1

for field_name, counts in sorted(meta_field_types.items()):
    if len(counts) > 1:
        print(f"\n*** INCONSISTENT METADATA FIELD: {field_name} ***")
        print(f"    Types: {dict(counts)}")
    else:
        tname, tcount = list(counts.items())[0]
        print(f"  metadata.{field_name:25s} {tname:15s} (all {tcount})")

# --- AUDIT 5: raw_json types ---
print("\n" + "=" * 60)
print("AUDIT 5: raw_json top-level fields")
print("=" * 60)

rj_field_types: dict[str, Counter] = {}
for doc in docs:
    raw = doc.model_dump()
    rj = raw.get("raw_json", {})
    if isinstance(rj, dict):
        for k, v in rj.items():
            if k not in rj_field_types:
                rj_field_types[k] = Counter()
            rj_field_types[k][type(v).__name__] += 1

for field_name, counts in sorted(rj_field_types.items()):
    if len(counts) > 1:
        print(f"\n*** INCONSISTENT RAW_JSON FIELD: {field_name} ***")
        print(f"    Types: {dict(counts)}")
        # Find a sample
        for doc in docs:
            rj = doc.model_dump().get("raw_json", {})
            if isinstance(rj, dict) and field_name in rj:
                print(f"    Sample value: {repr(rj[field_name])[:120]}")
                print(f"    Sample document_id: {doc.document_id}")
                break
    else:
        tname, tcount = list(counts.items())[0]
        print(f"  raw_json.{field_name:25s} {tname:15s} (all {tcount})")

print("\n=== SCHEMA AUDIT COMPLETE ===")
