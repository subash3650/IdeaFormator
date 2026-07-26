"""Minimal PlayStore test with schema audit."""
from pain_intelligence.ingestion.collectors.playstore.collector import PlayStoreCollector
from pain_intelligence.ingestion.collectors.playstore.adapter import PlayStoreAdapter
from pain_intelligence.ingestion.models import RawDocument
from pain_intelligence.ingestion.pipeline.persist import PersistStage, _audit_field_types, _log_type_inconsistencies
from pain_intelligence.ingestion.config import CollectorConfig
from pain_intelligence.ingestion.clients.base import HttpClient
from pain_intelligence.ingestion.pipeline.normalize import NormalizeStage
from pain_intelligence.ingestion.pipeline.validate import ValidateStage
from pain_intelligence.ingestion.pipeline.enrich import EnrichStage
from pathlib import Path
import tempfile

# Use a minimal config
config = CollectorConfig(
    batch_size=100,
    max_pages=1,
    rate_limit=1.0,
    timeout=30,
    retry_count=3,
    retry_delay=2.0,
    language="en",
    country="us",
    review_limit=20,
    sort="newest",
    apps=["com.openai.chatgpt"],
    apps_config_path="",
)

# Create collector
client = HttpClient(timeout=30, retry_count=3, retry_delay=2.0)
collector = PlayStoreCollector(config, client)

# Collect
collector.authenticate()
all_docs = []
for batch in collector.fetch(state=None):
    print(f"Batch: {len(batch)} items")

    # Transform
    transformed = collector._adapter.transform_batch(batch)
    print(f"  Transformed: {len(transformed)} records")

    # Normalize
    normalize = NormalizeStage()
    docs = normalize.run(transformed)
    print(f"  Normalized: {len(docs)} documents")

    # Validate
    validate = ValidateStage()
    valid, invalid = validate.run(docs)
    print(f"  Valid: {len(valid)}, Invalid: {len(invalid)}")

    # Enrich
    enrich = EnrichStage()
    enriched = enrich.run(valid)
    print(f"  Enriched: {len(enriched)} documents")

    all_docs.extend(enriched)

print(f"\nTotal documents: {len(all_docs)}")

# Schema audit before persistence
print("\n=== Schema Audit ===")
field_types = _audit_field_types(all_docs)
_log_type_inconsistencies(field_types, all_docs)

# Show all fields and their types
print("\n=== Field Type Summary ===")
for field_name, counts in sorted(field_types.items()):
    print(f"{field_name}: {dict(counts)}")

# Show first few docs' problematic fields
if all_docs:
    print("\n=== First doc to_flat_dict ===")
    flat = all_docs[0].to_flat_dict()
    for k, v in flat.items():
        print(f"  {k}: {type(v).__name__} = {repr(v)[:80]}")
