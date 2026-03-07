"""Document chunker with overlap and metadata tagging."""

import re
import hashlib
import logging
from typing import List, Dict, Any

from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentChunker:
    """Split parsed document content into overlapping, metadata-tagged chunks."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        max_chunks: int = None,
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.max_chunks = max_chunks or settings.MAX_CHUNKS_PER_DOC

    def chunk_document(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Turn a parsed-document dict into a list of chunk dicts.

        Each chunk dict:
          - chunk_id: deterministic hash
          - text: the chunk content
          - modality: "text" | "table" | "form"
          - source: original filename
          - chunk_index: position within this document
          - metadata: extra info
        """
        source = parsed.get("source", "unknown")
        chunks: List[Dict[str, Any]] = []

        # 1. Text chunks (sentence-aware sliding window)
        text_blocks = parsed.get("text_blocks", [])
        if text_blocks:
            full_text = "\n".join(text_blocks)
            text_chunks = self._sliding_window_chunk(full_text)
            for i, tc in enumerate(text_chunks):
                chunks.append(self._make_chunk(
                    text=tc,
                    modality="text",
                    source=source,
                    chunk_index=len(chunks),
                    extra={"text_chunk_pos": i},
                ))

        # 2. Table chunks
        tables = parsed.get("tables", [])
        for i, table_md in enumerate(tables):
            header = f"[TABLE {i + 1}]\n"
            chunks.append(self._make_chunk(
                text=header + table_md,
                modality="table",
                source=source,
                chunk_index=len(chunks),
                extra={"table_index": i},
            ))

        # 3. Key-value / form chunks (group in batches)
        kv_pairs = parsed.get("key_value_pairs", [])
        if kv_pairs:
            kv_text_parts = [f"{kv['key']}: {kv['value']}" for kv in kv_pairs]
            kv_full = "\n".join(kv_text_parts)
            kv_chunks = self._sliding_window_chunk(kv_full)
            for i, kvc in enumerate(kv_chunks):
                header = "[FORM DATA]\n"
                chunks.append(self._make_chunk(
                    text=header + kvc,
                    modality="form",
                    source=source,
                    chunk_index=len(chunks),
                    extra={"form_chunk_pos": i},
                ))

        # Enforce max chunks
        if len(chunks) > self.max_chunks:
            logger.warning(
                "Document %s produced %d chunks, truncating to %d",
                source, len(chunks), self.max_chunks,
            )
            chunks = chunks[: self.max_chunks]

        logger.info("Chunked %s: %d chunks", source, len(chunks))
        return chunks



    def _sliding_window_chunk(self, text: str) -> List[str]:
        """Sentence-aware sliding-window chunking."""
        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks = []
        current: List[str] = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_len + sent_len > self.chunk_size and current:
                chunk_text = " ".join(current).strip()
                if chunk_text:
                    chunks.append(chunk_text)

                # Overlap: keep last N characters worth of sentences
                overlap_budget = self.chunk_overlap
                overlap_sentences: List[str] = []
                overlap_len = 0
                for s in reversed(current):
                    if overlap_len + len(s) > overlap_budget:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_len += len(s)

                current = overlap_sentences
                current_len = overlap_len

            current.append(sentence)
            current_len += sent_len

        # Final chunk
        if current:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)

        return chunks

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex heuristic."""
        # Split on sentence-ending punctuation followed by space or newline
        raw = re.split(r"(?<=[.!?])\s+|\n+", text)
        return [s.strip() for s in raw if s.strip()]


    @staticmethod
    def _make_chunk(
        text: str,
        modality: str,
        source: str,
        chunk_index: int,
        extra: Dict = None,
    ) -> Dict[str, Any]:
        chunk_id = hashlib.sha256(f"{source}:{chunk_index}:{text[:64]}".encode()).hexdigest()[:16]
        return {
            "chunk_id": chunk_id,
            "text": text,
            "modality": modality,
            "source": source,
            "chunk_index": chunk_index,
            "metadata": extra or {},
        }
