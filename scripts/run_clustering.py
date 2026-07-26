"""Run clustering and show results."""
import sys
sys.stdout.reconfigure(line_buffering=True)
import json
from phase2.clustering.config import load_clustering_config
from phase2.clustering.engine import ClusteringEngine

cfg = load_clustering_config("configs/default.yaml")
engine = ClusteringEngine(cfg)
result = engine.generate(force=True)
print(json.dumps(result, indent=2))
