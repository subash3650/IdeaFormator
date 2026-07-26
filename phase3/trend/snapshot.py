"""Trend snapshot system — capture, scan, and compare pipeline snapshots."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

from phase3.trend.schema import TrendSnapshot


SNAPSHOT_DIR_NAME = "snapshots"
SNAPSHOT_INDEX = ".snapshot_index.json"
SNAPSHOT_MANIFEST = "snapshot_manifest.json"


# ---------------------------------------------------------------------------
# Snapshot Scanner
# ---------------------------------------------------------------------------


class TrendSnapshotScanner:
    """Scans the snapshots directory and returns sorted snapshot metadata."""

    def __init__(self, knowledge_dir: Path) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._snapshots_dir = self._knowledge_dir / SNAPSHOT_DIR_NAME

    @property
    def snapshots_dir(self) -> Path:
        return self._snapshots_dir

    def scan(self) -> list[TrendSnapshot]:
        if not self._snapshots_dir.exists():
            return []
        snapshots: list[TrendSnapshot] = []
        for entry in sorted(self._snapshots_dir.iterdir()):
            if not entry.is_dir():
                continue
            manifest_path = entry / SNAPSHOT_MANIFEST
            if not manifest_path.exists():
                continue
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                snapshots.append(TrendSnapshot(**data))
            except Exception:
                continue
        return sorted(snapshots, key=lambda s: s.timestamp)

    def count(self) -> int:
        return len(self.scan())

    def latest(self) -> TrendSnapshot | None:
        all_snaps = self.scan()
        return all_snaps[-1] if all_snaps else None

    def prior(self, window: int = 1) -> TrendSnapshot | None:
        all_snaps = self.scan()
        if len(all_snaps) < window + 1:
            return None
        return all_snaps[-(window + 1)]


# ---------------------------------------------------------------------------
# Snapshot Loader
# ---------------------------------------------------------------------------


class TrendSnapshotLoader:
    """Loads asset data from a specific snapshot."""

    ASSETS = [
        "observations",
        "evidence",
        "problem_signals",
        "semantic_relationships",
        "semantic_clusters",
        "knowledge_graph",
        "reasoning_chains",
        "root_causes",
        "opportunities",
    ]

    def __init__(self, knowledge_dir: Path) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._snapshots_dir = self._knowledge_dir / SNAPSHOT_DIR_NAME

    def load_asset(self, snapshot: TrendSnapshot, asset_name: str) -> pl.DataFrame:
        snapshot_dir = self._snapshots_dir / snapshot.run_id
        path = snapshot_dir / "assets" / f"{asset_name}.parquet"
        if not path.exists():
            return pl.DataFrame()
        try:
            return pl.read_parquet(path)
        except Exception:
            return pl.DataFrame()

    def load_observations(self, snapshot: TrendSnapshot) -> pl.DataFrame:
        return self.load_asset(snapshot, "observations")

    def load_evidence(self, snapshot: TrendSnapshot) -> pl.DataFrame:
        return self.load_asset(snapshot, "evidence")

    def load_signals(self, snapshot: TrendSnapshot) -> pl.DataFrame:
        return self.load_asset(snapshot, "problem_signals")

    def load_opportunities(self, snapshot: TrendSnapshot) -> pl.DataFrame:
        return self.load_asset(snapshot, "opportunities")

    def load_root_causes(self, snapshot: TrendSnapshot) -> pl.DataFrame:
        return self.load_asset(snapshot, "root_causes")


# ---------------------------------------------------------------------------
# Snapshot Comparator
# ---------------------------------------------------------------------------


@pl.api.register_lazyframe_namespace("trend")
class SnapshotDelta:
    """Result of comparing two snapshots — transient, not persisted."""

    def __init__(self) -> None:
        self.observation_count_before: int = 0
        self.observation_count_after: int = 0
        self.evidence_count_before: int = 0
        self.evidence_count_after: int = 0
        self.signal_count_before: int = 0
        self.signal_count_after: int = 0
        self.opportunity_count_before: int = 0
        self.opportunity_count_after: int = 0
        self.entity_counts_before: dict[str, int] = {}
        self.entity_counts_after: dict[str, int] = {}

    @property
    def observation_growth_pct(self) -> float:
        if self.observation_count_before == 0:
            return 0.0
        return ((self.observation_count_after - self.observation_count_before) / self.observation_count_before) * 100

    @property
    def evidence_growth_pct(self) -> float:
        if self.evidence_count_before == 0:
            return 0.0
        return ((self.evidence_count_after - self.evidence_count_before) / self.evidence_count_before) * 100


class TrendSnapshotComparator:
    """Compares two snapshots and produces a delta."""

    def __init__(self, knowledge_dir: Path) -> None:
        self._loader = TrendSnapshotLoader(knowledge_dir)

    def compare(self, snapshot_a: TrendSnapshot, snapshot_b: TrendSnapshot) -> SnapshotDelta:
        delta = SnapshotDelta()

        obs_a = self._loader.load_observations(snapshot_a)
        obs_b = self._loader.load_observations(snapshot_b)
        delta.observation_count_before = len(obs_a)
        delta.observation_count_after = len(obs_b)

        ev_a = self._loader.load_evidence(snapshot_a)
        ev_b = self._loader.load_evidence(snapshot_b)
        delta.evidence_count_before = len(ev_a)
        delta.evidence_count_after = len(ev_b)

        sig_a = self._loader.load_signals(snapshot_a)
        sig_b = self._loader.load_signals(snapshot_b)
        delta.signal_count_before = len(sig_a)
        delta.signal_count_after = len(sig_b)

        opp_a = self._loader.load_opportunities(snapshot_a)
        opp_b = self._loader.load_opportunities(snapshot_b)
        delta.opportunity_count_before = len(opp_a)
        delta.opportunity_count_after = len(opp_b)

        # Entity-level comparison by computing counts from observations
        delta.entity_counts_before = self._compute_entity_counts(obs_a)
        delta.entity_counts_after = self._compute_entity_counts(obs_b)

        return delta

    def _compute_entity_counts(self, df: pl.DataFrame) -> dict[str, int]:
        counts: dict[str, int] = {}
        if df.height == 0:
            return counts
        for col in ["entity_type", "entity_id", "source_entity"]:
            if col in df.columns:
                for val in df[col].drop_nulls().unique():
                    key = str(val)
                    c = df.filter(pl.col(col) == val).height
                    counts[key] = counts.get(key, 0) + c
        return counts


# ---------------------------------------------------------------------------
# Snapshot Builder
# ---------------------------------------------------------------------------


class TrendSnapshotBuilder:
    """Creates snapshots by copying current assets into a versioned directory."""

    ASSET_FILES = [
        "observations.parquet",
        "evidence.parquet",
        "problem_signals.parquet",
        "semantic_relationships.parquet",
        "semantic_clusters.parquet",
        "knowledge_graph.parquet",
        "reasoning_chains.parquet",
        "root_causes.parquet",
        "opportunities.parquet",
    ]

    def __init__(self, knowledge_dir: Path) -> None:
        self._knowledge_dir = Path(knowledge_dir)
        self._snapshots_dir = self._knowledge_dir / SNAPSHOT_DIR_NAME

    def create(self, run_id: str) -> TrendSnapshot:
        snapshot_dir = self._snapshots_dir / run_id
        assets_dir = snapshot_dir / "assets"
        assets_dir.mkdir(parents=True, exist_ok=True)

        checksums: dict[str, str] = {}
        sizes: dict[str, int] = {}
        counts: dict[str, int] = {}

        source_assets = self._knowledge_dir / "assets"

        for filename in self.ASSET_FILES:
            src = source_assets / filename
            dst = assets_dir / filename
            if src.exists():
                shutil.copy2(str(src), str(dst))
                h = hashlib.sha256()
                data = src.read_bytes()
                h.update(data)
                checksums[filename] = h.hexdigest()[:16]
                sizes[filename] = len(data)
                try:
                    df = pl.read_parquet(src)
                    counts[filename] = len(df)
                except Exception:
                    counts[filename] = 0
            else:
                checksums[filename] = ""
                sizes[filename] = 0
                counts[filename] = 0

        raw = f"{run_id}-{datetime.now(timezone.utc).isoformat()}"
        snapshot_id = hashlib.sha256(raw.encode()).hexdigest()[:16]

        snapshot = TrendSnapshot(
            snapshot_id=snapshot_id,
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            observation_count=counts.get("observations.parquet", 0),
            evidence_count=counts.get("evidence.parquet", 0),
            signal_count=counts.get("problem_signals.parquet", 0),
            opportunity_count=counts.get("opportunities.parquet", 0),
            asset_checksums=checksums,
            asset_sizes=sizes,
        )

        manifest_path = snapshot_dir / SNAPSHOT_MANIFEST
        manifest_path.write_text(
            json.dumps(snapshot.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

        self._update_index(snapshot)

        return snapshot

    def _update_index(self, snapshot: TrendSnapshot) -> None:
        index_path = self._snapshots_dir / SNAPSHOT_INDEX
        entries: list[dict[str, Any]] = []
        if index_path.exists():
            try:
                entries = json.loads(index_path.read_text(encoding="utf-8"))
            except Exception:
                entries = []
        entries.append(snapshot.model_dump(mode="json"))
        index_path.write_text(
            json.dumps(entries, indent=2, default=str), encoding="utf-8"
        )
