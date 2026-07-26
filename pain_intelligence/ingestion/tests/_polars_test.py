"""Test Polars DataFrame creation with mixed types."""
import polars as pl

# Test 1: str + None values
rows = [
    {"id": "1", "created_at": "2026-01-15T00:00:00+00:00"},
    {"id": "2", "created_at": None},
    {"id": "3", "created_at": "2026-03-10T10:00:00+00:00"},
]

print("Test 1: str + None")
df = pl.DataFrame(rows)
print(f"  Schema: {df.schema}")
print(f"  created_at dtype: {df['created_at'].dtype}")

# Test 2: Add explicit schema
print("\nTest 2: With explicit schema")
schema = {"id": pl.Utf8, "created_at": pl.Utf8}
df2 = pl.DataFrame(rows, schema=schema)
print(f"  Schema: {df2.schema}")

# Test 3: Use polars from_dicts
print("\nTest 3: from_dicts")
df3 = pl.from_dicts(rows)
print(f"  Schema: {df3.schema}")

# Test 4: What if we have datetime objects mixed with strings?
from datetime import datetime, timezone
rows_mixed = [
    {"id": "1", "ts": datetime(2026, 1, 15, tzinfo=timezone.utc)},
    {"id": "2", "ts": "2026-02-20T10:00:00+00:00"},
    {"id": "3", "ts": None},
]

print("\nTest 4: datetime + str + None (SHOULD FAIL)")
try:
    df4 = pl.DataFrame(rows_mixed)
    print(f"  Schema: {df4.schema}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test 5: What about dict values?
rows_dict = [
    {"id": "1", "metadata": '{"key": "value"}'},
    {"id": "2", "metadata": "{'key': 'value2'}"},
]

print("\nTest 5: String dict representations")
df5 = pl.DataFrame(rows_dict)
print(f"  Schema: {df5.schema}")
print(f"  metadata dtype: {df5['metadata'].dtype}")
