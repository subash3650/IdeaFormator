"""Chunk processor for memory-efficient document processing.

Streams documents through the preprocessing pipeline in configurable
chunks, never loading the entire dataset into memory.
"""

from __future__ import annotations

from typing import Any, Iterator

import polars as pl

from pain_intelligence.loaders.base import BaseLoader
from pain_intelligence.preprocessing.duplicate_detector import DuplicateDetector
from pain_intelligence.preprocessing.language_detector import LanguageDetector
from pain_intelligence.preprocessing.base import TextCleanerProtocol
from pain_intelligence.schema.document import Document, RemovalReason, RemovedDocument
from pain_intelligence.logging_config.logger import get_logger

logger = get_logger()


class ChunkProcessor:
    """Process documents through the cleaning pipeline in chunks.

    Args:
        cleaners: Ordered list of text cleaners to apply.
        duplicate_detector: Optional dedup detector.
        language_detector: Optional language detector.
        min_document_length: Minimum character length for a document.
        max_document_length: Maximum character length for a document.
        supported_languages: List of allowed language codes.
    """

    def __init__(
        self,
        cleaners: list[TextCleanerProtocol],
        duplicate_detector: DuplicateDetector | None = None,
        language_detector: LanguageDetector | None = None,
        min_document_length: int = 10,
        max_document_length: int = 100_000,
        supported_languages: list[str] | None = None,
    ) -> None:
        self._cleaners = cleaners
        self._dup_detector = duplicate_detector
        self._lang_detector = language_detector
        self._min_len = min_document_length
        self._max_len = max_document_length
        self._supported_langs = supported_languages or []

    def process_chunks(
        self,
        chunks: Iterator[pl.DataFrame],
        loader: BaseLoader,
    ) -> Iterator[tuple[list[Document], list[RemovedDocument]]]:
        """Process DataFrame chunks through the full pipeline.

        Yields:
            Tuple of (processed documents, removed documents) per chunk.
        """
        for chunk_num, chunk in enumerate(chunks):
            logger.info("Processing chunk {} ({} rows)", chunk_num + 1, len(chunk))

            docs = loader.transform_chunk(chunk)
            processed: list[Document] = []
            removed: list[RemovedDocument] = []

            for doc in docs:
                result = self._process_document(doc)
                if result is not None:
                    if isinstance(result, RemovedDocument):
                        removed.append(result)
                    else:
                        processed.append(result)

            logger.info(
                "Chunk {}: {} processed, {} removed",
                chunk_num + 1,
                len(processed),
                len(removed),
            )
            yield processed, removed

    def _process_document(self, doc: Document) -> Document | RemovedDocument | None:
        """Process a single document through the pipeline.

        Returns:
            Document if valid, RemovedDocument if removed, or None on error.
        """
        text = doc.text

        try:
            for cleaner in self._cleaners:
                if hasattr(cleaner, "clean"):
                    text = cleaner.clean(text)
        except Exception as e:
            logger.warning("Cleaning error on doc {}: {}", doc.id, e)
            return RemovedDocument(
                document_id=doc.id,
                platform=doc.platform,
                source_dataset=doc.source_dataset,
                text_preview=text[:100] if text else "",
                reason=RemovalReason.ENCODING_ERROR,
                original_length=doc.document_length,
            )

        if not text or not text.strip():
            return self._make_removed(doc, RemovalReason.EMPTY_TEXT, text)

        if len(text.strip()) < self._min_len:
            return self._make_removed(doc, RemovalReason.TOO_SHORT, text)

        if len(text) > self._max_len:
            return self._make_removed(doc, RemovalReason.TOO_LONG, text)

        if self._dup_detector is not None:
            if self._dup_detector.is_duplicate(text, doc.id):
                return self._make_removed(doc, RemovalReason.DUPLICATE, text)

        if self._lang_detector is not None:
            detected_lang = self._lang_detector.detect_language(text)
            doc.language = detected_lang
            if not self._lang_detector.is_supported(detected_lang):
                return self._make_removed(
                    doc, RemovalReason.UNSUPPORTED_LANGUAGE, text
                )

        doc.clean_text = text
        doc.document_length = len(text)
        return doc

    @staticmethod
    def _make_removed(
        doc: Document, reason: RemovalReason, text: str
    ) -> RemovedDocument:
        """Create a RemovedDocument record."""
        return RemovedDocument(
            document_id=doc.id,
            platform=doc.platform,
            source_dataset=doc.source_dataset,
            text_preview=text[:100] if text else "",
            reason=reason,
            original_length=doc.document_length,
        )
