"""RAG Pipeline. Orchestrates ingest and query flows."""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path

from config.settings import settings
from ingestion.document_parser import DocumentParser
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.generator import AnswerGenerator

logger = logging.getLogger(__name__)


def _create_vector_client():
    """Create the appropriate vector store client based on settings."""
    if settings.VECTOR_BACKEND == "faiss":
        from indexing.faiss_client import FAISSClient
        return FAISSClient(dimension=settings.EMBEDDING_DIMENSION)
    else:
        from indexing.opensearch_client import OpenSearchClient
        return OpenSearchClient()


class RAGPipeline:
    """
    End-to-end Multimodal RAG pipeline.

    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest("document.pdf", file_bytes)
        result = pipeline.query("What is the revenue for Q3?")
    """

    def __init__(self, lazy_init: bool = False):
        """
        Initialize all pipeline components.

        Args:
            lazy_init: if True, defer heavy model loading until first use
        """
        self._parser: Optional[DocumentParser] = None
        self._chunker: Optional[DocumentChunker] = None
        self._embedder: Optional[Embedder] = None
        self._os_client: Optional[OpenSearchClient] = None
        self._retriever: Optional[Retriever] = None
        self._reranker: Optional[Reranker] = None
        self._generator: Optional[AnswerGenerator] = None

        if not lazy_init:
            self._init_all()

    def _init_all(self):
        self.parser
        self.chunker
        self.embedder
        self.os_client
        self.retriever
        self.reranker
        self.generator

    # Lazy-loaded components

    @property
    def parser(self) -> DocumentParser:
        if self._parser is None:
            self._parser = DocumentParser()
        return self._parser

    @property
    def chunker(self) -> DocumentChunker:
        if self._chunker is None:
            self._chunker = DocumentChunker()
        return self._chunker

    @property
    def embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def os_client(self):
        if self._os_client is None:
            self._os_client = _create_vector_client()
        return self._os_client

    @property
    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever(
                embedder=self.embedder,
                os_client=self.os_client,
            )
        return self._retriever

    @property
    def reranker(self) -> Reranker:
        if self._reranker is None:
            self._reranker = Reranker()
        return self._reranker

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = AnswerGenerator()
        return self._generator



    def ingest(
        self,
        filename: str,
        file_bytes: bytes,
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        """
        Full ingestion pipeline: parse → chunk → embed → index.

        Args:
            filename: original filename
            file_bytes: raw file content
            replace_existing: delete old chunks for this source first

        Returns:
            Dict with ingestion stats
        """
        logger.info("Ingesting: %s (%.1f KB)", filename, len(file_bytes) / 1024)

        # Ensure index exists
        self.os_client.create_index(dimension=self.embedder.embedding_dimension)

        # Optionally remove old chunks
        if replace_existing:
            deleted = self.os_client.delete_by_source(filename)
            if deleted:
                logger.info("Removed %d existing chunks for %s", deleted, filename)

        # Step 1: Parse
        parsed = self.parser.parse(filename, file_bytes)

        # Step 2: Chunk
        chunks = self.chunker.chunk_document(parsed)

        # Step 3: Embed
        chunks = self.embedder.embed_chunks(chunks)

        # Step 4: Index
        indexed = self.os_client.index_chunks(chunks)

        result = {
            "filename": filename,
            "num_text_blocks": parsed["metadata"]["num_text_blocks"],
            "num_tables": parsed["metadata"]["num_tables"],
            "num_chunks": len(chunks),
            "num_indexed": indexed,
            "file_size_bytes": len(file_bytes),
        }

        logger.info("Ingestion complete: %s, %d chunks indexed", filename, indexed)
        return result



    def query(
        self,
        question: str,
        top_k: int = None,
        rerank_top_k: int = None,
        search_mode: str = "hybrid",
        source_filter: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Full query pipeline: retrieve, rerank, generate answer."""
        top_k = top_k or settings.TOP_K_RESULTS
        rerank_top_k = rerank_top_k or settings.RERANK_TOP_K

        logger.info("Query: %s", question[:80])

        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=top_k,
            mode=search_mode,
            source_filter=source_filter,
        )

        # Step 2: Rerank
        reranked = self.reranker.rerank(
            query=question,
            chunks=retrieved,
            top_k=rerank_top_k,
        )

        # Step 3: Generate
        gen_result = self.generator.generate(question, reranked)
        gen_result["retrieved_chunks"] = reranked
        return gen_result



    def get_status(self) -> Dict[str, Any]:
        """Return system status."""
        health = self.os_client.health_check()
        return {
            "opensearch": health,
            "total_chunks": self.os_client.get_doc_count(),
            "indexed_sources": self.os_client.get_sources(),
            "embedding_model": settings.EMBEDDING_MODEL,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
            "llm_model": AnswerGenerator.MODEL,
        }

    def delete_source(self, source: str) -> int:
        """Delete all chunks for a source document."""
        return self.os_client.delete_by_source(source)

    def reset_index(self):
        """Delete and recreate the index."""
        self.os_client.delete_index()
        self.os_client.create_index(dimension=self.embedder.embedding_dimension)
        logger.info("Index reset complete")
