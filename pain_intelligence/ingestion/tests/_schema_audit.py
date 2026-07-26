"""Quick schema audit script."""
from pain_intelligence.ingestion.collectors.playstore.adapter import PlayStoreAdapter
from pain_intelligence.ingestion.pipeline.persist import _audit_field_types, _log_type_inconsistencies
from pain_intelligence.ingestion.models import RawDocument
from datetime import datetime, timezone

adapter = PlayStoreAdapter()

# Simulate reviews with mixed timestamp types
reviews = [
    {"reviewId": "r1", "userName": "User1", "score": 5, "content": "Great app", "at": datetime(2026, 1, 15, tzinfo=timezone.utc)},
    {"reviewId": "r2", "userName": "User2", "score": 4, "content": "Good app", "at": "2026-02-20T10:00:00+00:00"},
    {"reviewId": "r3", "userName": "User3", "score": 3, "content": "OK app", "at": None},
    {"reviewId": "r4", "userName": "User4", "score": 2, "content": "Bad app", "at": 1719225600},
    {"reviewId": "r5", "userName": "User5", "score": 1, "content": "Terrible", "at": datetime.now(timezone.utc)},
]

# Transform
transformed = adapter.transform_batch(reviews)
print(f"Transformed {len(transformed)} records")

# Check types of created_at in adapter output
for i, t in enumerate(transformed):
    val = t["created_at"]
    print(f"Record {i}: created_at type = {type(val).__name__}, value = {val}")

# Create RawDocuments
docs = []
for t in transformed:
    doc = RawDocument(**t)
    docs.append(doc)

print(f"\nCreated {len(docs)} RawDocuments")

# Check types AFTER RawDocument creation
for i, doc in enumerate(docs):
    print(f"Doc {i}: created_at type = {type(doc.created_at).__name__}, value = {doc.created_at}")

# Check to_flat_dict output
print("\n--- to_flat_dict output ---")
for i, doc in enumerate(docs):
    flat = doc.to_flat_dict()
    print(f"Doc {i}: created_at type = {type(flat['created_at']).__name__}, value = {flat['created_at']!r}")

# Audit
print("\n--- Full audit ---")
field_types = _audit_field_types(docs)
_log_type_inconsistencies(field_types, docs)

# Show specific fields
for field in ["created_at", "updated_at", "ingested_at"]:
    if field in field_types:
        print(f"\n{field}: {dict(field_types[field])}")
