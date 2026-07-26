"""Diagnostic script for clustering results."""
import sys
sys.stdout.reconfigure(line_buffering=True)

from phase2.clustering.config import load_clustering_config
from phase2.clustering.engine import ClusteringEngine
from phase2.clustering.store import SemanticClusterStore
from collections import Counter

cfg = load_clustering_config("configs/default.yaml")
store = SemanticClusterStore(cfg.output_directory)
clusters = store.load()

sizes = [c.member_count for c in clusters]
large = [c for c in clusters if c.member_count > 500]
print(f"Total clusters: {len(clusters)}")
print(f"Total large clusters (>500): {len(large)}")
if sizes:
    print(f"Largest cluster size: {max(sizes)}")
    print(f"Top 5 largest: {sorted(sizes, reverse=True)[:5]}")

# Validation
graph_engine = ClusteringEngine(cfg)
result = graph_engine.verify()
print(f"Valid: {result['valid']}")
codes = Counter(i['code'] for i in result['issues'])
for code, count in codes.most_common(5):
    print(f"  {code}: {count}")
