"""Pipeline orchestrator.

Coordinates the entire data ingestion and preprocessing workflow:
1. Load config
2. Discover datasets
3. Auto-detect loaders
4. Run preprocessing pipeline
5. Write outputs (processed, removed, stats, reports)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger

from pain_intelligence.loaders.registry import get_loader_for_file
import pain_intelligence.loaders  # noqa: F401 — triggers @register_loader for all loaders
from pain_intelligence.preprocessing.duplicate_detector import DuplicateDetector
from pain_intelligence.preprocessing.language_detector import LanguageDetector
from pain_intelligence.preprocessing.encoding_fixer import EncodingFixer
from pain_intelligence.preprocessing.html_cleaner import HtmlCleaner
from pain_intelligence.preprocessing.unicode_normalizer import UnicodeNormalizer
from pain_intelligence.preprocessing.url_cleaner import UrlCleaner
from pain_intelligence.preprocessing.emoji_processor import EmojiProcessor
from pain_intelligence.preprocessing.whitespace_cleaner import WhitespaceCleaner
from pain_intelligence.preprocessing.text_normalizer import TextNormalizer
from pain_intelligence.preprocessing.spell_correction import SpellCorrectionInterface
from pain_intelligence.pipeline.chunk_processor import ChunkProcessor
from pain_intelligence.schema.document import Document, RemovedDocument
from pain_intelligence.utils.io import write_json, write_dataframe, ensure_directory
from pain_intelligence.utils.stats import compute_statistics
from pain_intelligence.utils.config import load_config, get_nested
from pain_intelligence.logging_config.logger import setup_logger


class Orchestrator:
    """Main pipeline orchestrator.

    Coordinates loading, preprocessing, and output generation.
    All configuration flows through the YAML config file.

    Args:
        config_path: Path to YAML configuration file.
    """

    def __init__(self, config_path: str = "configs/default.yaml") -> None:
        self.config = load_config(config_path)
        self._setup_logging()
        self._stats: dict[str, Any] = {
            "pipeline_start": None,
            "pipeline_end": None,
            "datasets_processed": [],
            "total_loaded": 0,
            "total_processed": 0,
            "total_removed": 0,
            "errors": [],
        }

    def _setup_logging(self) -> None:
        """Configure logging from config."""
        log_cfg = self.config.get("logging", {})
        setup_logger(
            level=get_nested(self.config, "logging", "level", default="INFO"),
            log_file=get_nested(self.config, "logging", "file", default="outputs/pipeline.log"),
            rotation=get_nested(self.config, "logging", "rotation", default="10 MB"),
            retention=get_nested(self.config, "logging", "retention", default="7 days"),
        )

    def run(self) -> dict[str, Any]:
        """Execute the full pipeline.

        Returns:
            Processing report dictionary.
        """
        start_time = time.time()
        self._stats["pipeline_start"] = time.strftime("%Y-%m-%dT%H:%M:%S")

        raw_dir = Path(get_nested(self.config, "paths", "raw_datasets_dir", default="Datasets"))
        outputs_dir = Path(get_nested(self.config, "paths", "outputs_dir", default="outputs"))
        ensure_directory(outputs_dir)

        proc_cfg = self.config.get("processing", {})
        pre_cfg = self.config.get("preprocessing", {})
        out_cfg = self.config.get("output", {})

        chunk_size = get_nested(self.config, "processing", "chunk_size", default=50_000)
        min_doc_len = get_nested(self.config, "processing", "min_document_length", default=10)
        max_doc_len = get_nested(pre_cfg, "max_document_length", default=100_000)
        supported_langs = get_nested(self.config, "processing", "supported_languages", default=["en"])

        cleaners = self._build_cleaners(pre_cfg)
        dup_detector = DuplicateDetector() if pre_cfg.get("remove_duplicates", True) else None
        lang_detector = (
            LanguageDetector(supported_languages=supported_langs)
            if pre_cfg.get("detect_language", True)
            else None
        )

        processor = ChunkProcessor(
            cleaners=cleaners,
            duplicate_detector=dup_detector,
            language_detector=lang_detector,
            min_document_length=min_doc_len,
            max_document_length=max_doc_len,
            supported_languages=supported_langs,
        )

        all_documents: list[Document] = []
        all_removed: list[dict[str, Any]] = []
        removed_models: list[RemovedDocument] = []

        dataset_files = sorted(raw_dir.glob("*.csv"))
        logger.info("Found {} CSV datasets in {}", len(dataset_files), raw_dir)

        for file_path in dataset_files:
            try:
                loader = get_loader_for_file(file_path, chunk_size=chunk_size)
                total_loaded = 0
                total_processed = 0
                total_removed_count = 0

                chunks = loader.load(file_path, chunk_size=chunk_size)
                for processed_docs, removed_docs in processor.process_chunks(chunks, loader):
                    all_documents.extend(processed_docs)
                    removed_models.extend(removed_docs)
                    total_processed += len(processed_docs)
                    total_removed_count += len(removed_docs)
                    total_loaded += len(processed_docs) + len(removed_docs)

                self._stats["datasets_processed"].append({
                    "file": file_path.name,
                    "platform": loader.platform.value,
                    "loaded": total_loaded,
                    "processed": total_processed,
                    "removed": total_removed_count,
                })
                self._stats["total_loaded"] += total_loaded
                self._stats["total_processed"] += total_processed
                self._stats["total_removed"] += total_removed_count

                logger.info(
                    "Dataset '{}': loaded={}, processed={}, removed={}",
                    file_path.name,
                    total_loaded,
                    total_processed,
                    total_removed_count,
                )
            except Exception as e:
                logger.error("Error processing '{}': {}", file_path.name, e)
                self._stats["errors"].append({
                    "file": file_path.name,
                    "error": str(e),
                })

        logger.info("Total processed: {}, Total removed: {}", len(all_documents), len(removed_models))

        self._write_reports(all_documents, removed_models, outputs_dir)
        self._write_outputs(all_documents, removed_models, outputs_dir, out_cfg)

        if dup_detector:
            dup_detector.close()

        elapsed = time.time() - start_time
        self._stats["pipeline_end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._stats["elapsed_seconds"] = round(elapsed, 2)

        logger.info("Pipeline complete in {:.2f}s", elapsed)
        return self._stats

    def _build_cleaners(self, pre_cfg: dict[str, Any]) -> list:
        """Build the ordered list of cleaners from config."""
        cleaners = []
        if pre_cfg.get("fix_encoding", True):
            cleaners.append(EncodingFixer())
        if pre_cfg.get("remove_html", True):
            cleaners.append(HtmlCleaner())
        if pre_cfg.get("normalize_unicode", True):
            cleaners.append(UnicodeNormalizer())
        if pre_cfg.get("remove_urls", True):
            cleaners.append(UrlCleaner())
        if pre_cfg.get("process_emojis", True):
            cleaners.append(EmojiProcessor())
        if pre_cfg.get("normalize_whitespace", True):
            cleaners.append(WhitespaceCleaner())
        cleaners.append(TextNormalizer(
            lowercase=pre_cfg.get("lowercase", False),
        ))
        cleaners.append(SpellCorrectionInterface())
        return cleaners

    def _write_outputs(
        self,
        documents: list[Document],
        removed: list[RemovedDocument],
        outputs_dir: Path,
        out_cfg: dict[str, Any],
    ) -> None:
        """Write processed and removed documents to disk.

        Uses batched writing to avoid materializing all rows in memory at once.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        formats = out_cfg.get("formats", ["parquet", "csv"])

        if documents:
            logger.info("Writing {} processed documents...", len(documents))
            self._write_documents_batched(documents, outputs_dir, formats)

        if removed:
            logger.info("Writing {} removed documents...", len(removed))
            removed_dicts = [r.model_dump() for r in removed]
            removed_df = pl.DataFrame(removed_dicts)
            write_dataframe(removed_df, outputs_dir / "removed.parquet", format="parquet")
            logger.info("Wrote {} removed records", len(removed))

    def _write_documents_batched(
        self,
        documents: list[Document],
        outputs_dir: Path,
        formats: list[str],
    ) -> None:
        """Write documents in batches to avoid memory spikes."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        unified_schema = pa.schema([
            ("id", pa.string()),
            ("platform", pa.string()),
            ("source_dataset", pa.string()),
            ("title", pa.string()),
            ("text", pa.string()),
            ("rating", pa.float64()),
            ("author", pa.string()),
            ("country", pa.string()),
            ("location", pa.string()),
            ("language", pa.string()),
            ("created_at", pa.string()),
            ("metadata", pa.string()),
            ("raw_record", pa.string()),
            ("clean_text", pa.string()),
            ("document_length", pa.int64()),
        ])

        batch_size = 50_000
        pyarrow_batches: list[pa.RecordBatch] = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            flat_dicts = [d.to_flat_dict() for d in batch]
            table = pa.Table.from_pylist(flat_dicts, schema=unified_schema)
            pyarrow_batches.append(table.to_batches()[0])

        combined = pa.Table.from_batches(pyarrow_batches, schema=unified_schema)

        if "parquet" in formats:
            out_path = outputs_dir / "processed.parquet"
            pq.write_table(combined, str(out_path))
            logger.info("Wrote {}", out_path)

        if "csv" in formats:
            out_path = outputs_dir / "processed.csv"
            import pyarrow.csv as pcsv
            pcsv.write_csv(combined, str(out_path))
            logger.info("Wrote {}", out_path)

    def _write_reports(
        self,
        documents: list[Document],
        removed: list[RemovedDocument],
        outputs_dir: Path,
    ) -> None:
        """Write processing report and statistics."""
        removed_dicts = [r.model_dump() for r in removed]
        stats = compute_statistics(documents, removed_dicts)
        write_json(stats, outputs_dir / "dataset_statistics.json")
        logger.info("Wrote dataset_statistics.json")

        report = {
            "pipeline_version": "0.1.0",
            "processing_summary": self._stats,
            "dataset_statistics": stats,
        }
        write_json(report, outputs_dir / "processing_report.json")
        logger.info("Wrote processing_report.json")
