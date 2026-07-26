"""Pipeline integrity verification and quality dashboard generation.

Verifies:
  - All expected assets exist
  - No stale assets (run_id mismatch)
  - Schema consistency
  - Checksum matching
  - Embedding model consistency
  - Manifest consistency

Generates:
  - pipeline_dashboard.json
  - pipeline_dashboard.txt
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import yaml

from pain_intelligence.knowledge.exceptions import MissingAssetError, StaleAssetError
from pain_intelligence.knowledge.manifest import PIPELINE_VERSION, SCHEMA_VERSION, PipelineManifest, compute_checksum
from pain_intelligence.knowledge.metadata import get_run_id_from_asset, read_parquet_metadata


# Assets expected after each pipeline stage
EXPECTED_INFRASTRUCTURE_ASSETS = {
    "observations.parquet": "pain_intelligence/knowledge/assets/observations.parquet",
    "evidence.parquet": "pain_intelligence/knowledge/assets/evidence.parquet",
    "problem_signals.parquet": "pain_intelligence/knowledge/assets/problem_signals.parquet",
}

EXPECTED_PHASE2_ASSETS = {
    "embeddings_observation.parquet": "pain_intelligence/knowledge/assets/phase2/embeddings_observation.parquet",
    "embeddings_evidence.parquet": "pain_intelligence/knowledge/assets/phase2/embeddings_evidence.parquet",
    "embeddings_problem_signal.parquet": "pain_intelligence/knowledge/assets/phase2/embeddings_problem_signal.parquet",
    "semantic_relationships.parquet": "pain_intelligence/knowledge/assets/phase2/semantic_relationships.parquet",
    "semantic_clusters.parquet": "pain_intelligence/knowledge/assets/phase2/semantic_clusters.parquet",
}


def verify_pipeline(
    config_path: str = "configs/default.yaml",
    fix: bool = False,
) -> dict[str, Any]:
    """Run full pipeline integrity verification.

    Checks:
      - Missing assets
      - Stale assets (run_id mismatch)
      - Schema mismatch
      - Checksum mismatch
      - Manifest consistency
      - Embedding model consistency

    Returns a health report dict.
    """
    report: dict[str, Any] = {
        "overall": "PASS",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "checks": {},
        "warnings": [],
        "errors": [],
        "asset_details": {},
    }

    manifest = PipelineManifest("pain_intelligence/knowledge")
    manifest_data = manifest.to_dict()
    manifest_run_id = manifest_data.get("run_id", "")

    # Load config for source paths
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        report["errors"].append(f"Cannot load config: {e}")
        report["overall"] = "FAIL"
        return report

    # ── 1. Check knowledge base directory ──
    knowledge_dir = Path("pain_intelligence/knowledge")
    assets_dir = knowledge_dir / "assets"
    if not assets_dir.exists():
        report["errors"].append("Knowledge assets directory not found: pain_intelligence/knowledge/assets/")

    # ── 2. Check infrastructure assets (observations, evidence, problem_signals) ──
    for name, path_str in EXPECTED_INFRASTRUCTURE_ASSETS.items():
        check_result = _check_asset(
            path_str, name, manifest_run_id, manifest_data,
        )
        report["checks"][f"asset:{name}"] = check_result
        if check_result["status"] == "FAIL":
            report["errors"].append(f"{name}: {check_result['detail']}")
            report["overall"] = "FAIL"
        elif check_result["status"] == "WARN":
            report["warnings"].append(f"{name}: {check_result['detail']}")
            if report["overall"] == "PASS":
                report["overall"] = "WARN"
        report["asset_details"][name] = check_result

    # ── 3. Check phase 2 assets ──
    for name, path_str in EXPECTED_PHASE2_ASSETS.items():
        check_result = _check_asset(
            path_str, name, manifest_run_id, manifest_data,
        )
        report["checks"][f"asset:{name}"] = check_result
        if check_result["status"] == "FAIL":
            report["errors"].append(f"{name}: {check_result['detail']}")
            report["overall"] = "FAIL"
        elif check_result["status"] == "WARN":
            report["warnings"].append(f"{name}: {check_result['detail']}")
            if report["overall"] == "PASS":
                report["overall"] = "WARN"
        report["asset_details"][name] = check_result

    # ── 4. Check embedding model consistency ──
    embedding_dir = Path("pain_intelligence/knowledge/assets/phase2")
    model_consistency = _check_embedding_model_consistency(embedding_dir)
    report["checks"]["embedding_model_consistency"] = model_consistency
    if model_consistency["status"] == "FAIL":
        report["errors"].append(f"Embedding model inconsistency: {model_consistency['detail']}")
        report["overall"] = "FAIL"
    elif model_consistency["status"] == "WARN":
        report["warnings"].append(f"Embedding model: {model_consistency['detail']}")

    # ── 5. Check manifest ──
    manifest_check = _check_manifest(knowledge_dir)
    report["checks"]["manifest"] = manifest_check
    if manifest_check["status"] == "FAIL":
        report["errors"].append(f"Manifest: {manifest_check['detail']}")
        report["overall"] = "FAIL"
    elif manifest_check["status"] == "WARN":
        report["warnings"].append(f"Manifest: {manifest_check['detail']}")

    # ── 6. Check schema consistency ──
    for name, path_str in {**EXPECTED_INFRASTRUCTURE_ASSETS, **EXPECTED_PHASE2_ASSETS}.items():
        schema_check = _check_schema(path_str, name)
        if schema_check["status"] != "PASS":
            report["checks"][f"schema:{name}"] = schema_check
            if schema_check["status"] == "FAIL":
                report["errors"].append(f"Schema {name}: {schema_check['detail']}")
                report["overall"] = "FAIL"

    # ── 7. Count total checks ──
    passed = sum(1 for c in report["checks"].values() if c["status"] == "PASS")
    failed = sum(1 for c in report["checks"].values() if c["status"] == "FAIL")
    warned = sum(1 for c in report["checks"].values() if c["status"] == "WARN")
    report["summary"] = {
        "total_checks": len(report["checks"]),
        "passed": passed,
        "failed": failed,
        "warned": warned,
    }

    return report


def generate_dashboard(output_dir: str = "knowledge/reports") -> dict[str, Any]:
    """Generate pipeline quality dashboard (JSON + TXT)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    knowledge_dir = Path("pain_intelligence/knowledge")
    assets_dir = knowledge_dir / "assets"
    phase2_dir = assets_dir / "phase2"

    dash: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
    }

    # Documents (from infrastructure report or manifest)
    docs_count = _safe_count("outputs/processed.parquet")
    dash["documents"] = docs_count

    # Observations
    obs_count = _safe_count(assets_dir / "observations.parquet")
    dash["observation_count"] = obs_count
    dash["observation_extraction_rate"] = round(
        obs_count / max(docs_count, 1), 2
    ) if docs_count else 0

    # Evidence
    ev_count = _safe_count(assets_dir / "evidence.parquet")
    dash["evidence_count"] = ev_count
    dash["evidence_compression_ratio"] = round(
        obs_count / max(ev_count, 1), 2
    ) if ev_count else 0

    # Problem signals
    sig_count = _safe_count(assets_dir / "problem_signals.parquet")
    dash["problem_signal_count"] = sig_count
    dash["problem_signal_rate"] = round(
        sig_count / max(docs_count, 1), 4
    ) if docs_count else 0

    # Embeddings
    emb_counts = {
        "observation": _safe_count(phase2_dir / "embeddings_observation.parquet"),
        "evidence": _safe_count(phase2_dir / "embeddings_evidence.parquet"),
        "problem_signal": _safe_count(phase2_dir / "embeddings_problem_signal.parquet"),
    }
    dash["embedding_count"] = sum(emb_counts.values())
    dash["embeddings_by_source"] = emb_counts

    # Relationships
    rel_count = _safe_count(phase2_dir / "semantic_relationships.parquet")
    dash["relationship_count"] = rel_count

    # Average similarity
    rel_path = phase2_dir / "semantic_relationships.parquet"
    if rel_path.exists():
        try:
            rel_df = pl.read_parquet(str(rel_path))
            if rel_df.height > 0 and "similarity_score" in rel_df.columns:
                dash["average_similarity"] = round(float(rel_df["similarity_score"].mean()), 6)
            else:
                dash["average_similarity"] = 0.0
        except Exception:
            dash["average_similarity"] = 0.0
    else:
        dash["average_similarity"] = 0.0

    # Clusters
    cluster_path = phase2_dir / "semantic_clusters.parquet"
    cluster_count = _safe_count(cluster_path)
    dash["cluster_count"] = cluster_count
    dash["singleton_count"] = 0
    dash["average_cluster_size"] = 0.0
    dash["cluster_quality"] = 0.0

    if cluster_path.exists():
        try:
            cdf = pl.read_parquet(str(cluster_path))
            if cdf.height > 0:
                if "member_count" in cdf.columns:
                    dash["average_cluster_size"] = round(float(cdf["member_count"].mean()), 2)
                if "quality_score" in cdf.columns:
                    dash["cluster_quality"] = round(float(cdf["quality_score"].mean()), 4)
                if "member_count" in cdf.columns:
                    dash["singleton_count"] = int((cdf["member_count"] == 1).sum())
        except Exception:
            pass

    # Pipeline status
    manifest_path = knowledge_dir / "pipeline_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, encoding="utf-8") as f:
                m = json.load(f)
            dash["pipeline_run_id"] = m.get("run_id", "")
            dash["pipeline_started_at"] = m.get("started_at", "")
            dash["pipeline_completed_at"] = m.get("completed_at", "")

            started = m.get("started_at", "")
            completed = m.get("completed_at", "")
            if started and completed:
                try:
                    s = datetime.fromisoformat(started)
                    e = datetime.fromisoformat(completed)
                    diff = e - s
                    dash["pipeline_duration"] = str(diff)
                except Exception:
                    dash["pipeline_duration"] = "N/A"
            else:
                dash["pipeline_duration"] = "N/A"

            stages = m.get("stages", {})
            dash["stage_status"] = {
                s: info.get("status", "unknown") for s, info in stages.items()
            }

            all_success = all(
                info.get("status") == "completed" for info in stages.values()
            ) if stages else False
            dash["overall_status"] = "SUCCESS" if all_success else "INCOMPLETE"
        except Exception:
            dash["overall_status"] = "UNKNOWN"
    else:
        dash["overall_status"] = "NOT_STARTED"
        dash["pipeline_duration"] = "N/A"

    # Documents per source (if available)
    dash["documents_per_source"] = _get_source_counts()

    # Write JSON dashboard
    json_path = output_path / "pipeline_dashboard.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dash, f, indent=2, default=str, ensure_ascii=False)

    # Write text dashboard
    txt_path = output_path / "pipeline_dashboard.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("PAIN INTELLIGENCE — PIPELINE DASHBOARD\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {dash.get('generated_at', 'N/A')}\n")
        f.write(f"Pipeline Version: {dash.get('pipeline_version', 'N/A')}\n")
        f.write(f"Overall Status: {dash.get('overall_status', 'N/A')}\n\n")

        f.write("DOCUMENTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Documents:               {dash.get('documents', 0)}\n")
        f.write(f"  Documents per source:    {dash.get('documents_per_source', {})}\n\n")

        f.write("KNOWLEDGE EXTRACTION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Observations:            {dash.get('observation_count', 0)}\n")
        f.write(f"  Extraction Rate:         {dash.get('observation_extraction_rate', 0)}\n")
        f.write(f"  Evidence:                {dash.get('evidence_count', 0)}\n")
        f.write(f"  Compression Ratio:       {dash.get('evidence_compression_ratio', 0)}\n")
        f.write(f"  Problem Signals:         {dash.get('problem_signal_count', 0)}\n")
        f.write(f"  Signal Rate:             {dash.get('problem_signal_rate', 0)}\n\n")

        f.write("EMBEDDINGS & RELATIONSHIPS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Embedding Vectors:       {dash.get('embedding_count', 0)}\n")
        f.write(f"  Relationships:           {dash.get('relationship_count', 0)}\n")
        f.write(f"  Average Similarity:      {dash.get('average_similarity', 0)}\n\n")

        f.write("CLUSTERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Clusters:                {dash.get('cluster_count', 0)}\n")
        f.write(f"  Avg Cluster Size:        {dash.get('average_cluster_size', 0)}\n")
        f.write(f"  Cluster Quality:         {dash.get('cluster_quality', 0)}\n")
        f.write(f"  Singletons:              {dash.get('singleton_count', 0)}\n\n")

        f.write("PIPELINE\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Duration:                {dash.get('pipeline_duration', 'N/A')}\n")
        f.write(f"  Run ID:                  {dash.get('pipeline_run_id', 'N/A')}\n")
        if dash.get("stage_status"):
            f.write("  Stages:\n")
            for stage, status in dash["stage_status"].items():
                f.write(f"    {stage}: {status}\n")

    return dash


# ── Internal helpers ──────────────────────────────────────────────


def _check_asset(
    path_str: str,
    name: str,
    manifest_run_id: str,
    manifest_data: dict[str, Any],
) -> dict[str, str]:
    """Check a single asset's existence, run_id, and checksum."""
    p = Path(path_str)
    if not p.exists():
        return {"status": "FAIL", "detail": "asset file does not exist"}

    meta = read_parquet_metadata(p)
    asset_run_id = meta.get("run_id", "")

    # Check run_id consistency with manifest
    if manifest_run_id and asset_run_id and asset_run_id != manifest_run_id:
        return {
            "status": "FAIL",
            "detail": f"stale asset: run_id={asset_run_id}, manifest expects {manifest_run_id}",
        }

    # Check file is not empty
    try:
        df = pl.read_parquet(str(p), n_rows=1)
        has_data = df.height > 0
    except Exception:
        has_data = False

    size_bytes = p.stat().st_size
    checksum = compute_checksum(p)
    manifest_checksum = ""
    for asset_name, asset_info in manifest_data.get("assets", {}).items():
        if asset_name == name or asset_name == p.name or f"/{asset_name}" in str(p):
            manifest_checksum = asset_info.get("checksum", "")
            break

    detail_parts = []
    if not has_data:
        detail_parts.append("empty file")
    else:
        detail_parts.append(f"{size_bytes} bytes")

    if asset_run_id:
        detail_parts.append(f"run_id={asset_run_id}")
    if checksum:
        detail_parts.append(f"checksum={checksum}")

    if manifest_checksum and checksum and manifest_checksum != checksum:
        return {
            "status": "WARN",
            "detail": f"checksum mismatch: file={checksum}, manifest={manifest_checksum}",
        }

    return {"status": "PASS" if has_data else "WARN", "detail": "; ".join(detail_parts)}


def _check_schema(path_str: str, name: str) -> dict[str, str]:
    """Check that an asset's schema is valid."""
    p = Path(path_str)
    if not p.exists():
        return {"status": "SKIP", "detail": "file does not exist"}

    try:
        df = pl.read_parquet(str(p), n_rows=0)
        if df.width == 0:
            return {"status": "WARN", "detail": "no columns in schema"}
        return {"status": "PASS", "detail": f"{df.width} columns"}
    except Exception as e:
        return {"status": "FAIL", "detail": f"cannot read schema: {e}"}


def _check_manifest(knowledge_dir: Path) -> dict[str, str]:
    """Check manifest file integrity."""
    manifest_path = knowledge_dir / "pipeline_manifest.json"
    if not manifest_path.exists():
        return {"status": "WARN", "detail": "pipeline_manifest.json not found"}

    try:
        with open(manifest_path, encoding="utf-8") as f:
            m = json.load(f)
    except Exception as e:
        return {"status": "FAIL", "detail": f"cannot parse manifest: {e}"}

    if not m.get("run_id"):
        return {"status": "WARN", "detail": "manifest has no run_id"}
    if not m.get("assets"):
        return {"status": "WARN", "detail": "manifest has no assets"}
    return {"status": "PASS", "detail": f"{len(m.get('assets', {}))} assets recorded"}


def _check_embedding_model_consistency(dir_path: Path) -> dict[str, str]:
    """Check that all embedding files use the same model."""
    emb_files = list(dir_path.glob("embeddings_*.parquet"))
    if not emb_files:
        return {"status": "WARN", "detail": "no embedding files found"}

    models_seen: set[str] = set()
    for f in emb_files:
        meta = read_parquet_metadata(f)
        # Try to get model info from the data
        try:
            df = pl.read_parquet(str(f), n_rows=1)
            if df.height > 0 and "model" in df.columns:
                model_val = df["model"].to_list()[0]
                if model_val:
                    models_seen.add(str(model_val))
        except Exception:
            pass

    if len(models_seen) > 1:
        return {"status": "FAIL", "detail": f"multiple models found: {models_seen}"}
    if len(models_seen) == 1:
        return {"status": "PASS", "detail": f"consistent model: {next(iter(models_seen))}"}
    return {"status": "WARN", "detail": "no model information in embedding files"}


def _safe_count(path: str | Path) -> int:
    """Safely count rows in a Parquet file."""
    p = Path(path)
    if not p.exists():
        return 0
    try:
        df = pl.read_parquet(str(p))
        return df.height
    except Exception:
        return 0


def _get_source_counts() -> dict[str, int]:
    """Get document counts per source from processed data."""
    sources: dict[str, int] = {}

    # Try knowledge/processed
    for p in [Path("pain_intelligence/knowledge/processed/processed.parquet"), Path("outputs/processed.parquet")]:
        if p.exists():
            try:
                df = pl.read_parquet(str(p))
                if "platform" in df.columns:
                    source_counts = df["platform"].value_counts()
                    for row in source_counts.iter_rows(named=True):
                        platform = row.get("platform", "unknown")
                        count = row.get("count", row.get("counts", 0))
                        sources[str(platform)] = int(count)
                    break
            except Exception:
                pass
    return sources
