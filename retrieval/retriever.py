"""
Retriever — query embedding + hybrid search over the vector store.

Works with both OpenSearchClient and FAISSClient (duck-typed).
"""

import logging
from typing import List, Dict, Any, Optional

from config.settings import settings
from ingestion.embedder import Embedder

logger = logging.getLogger(__name__)


class Retriever:
    """Embeds user queries and retrieves relevant chunks from the vector store."""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        os_client=None,
    ):
        self.embedder = embedder or Embedder()
        self.os_client = os_client  # Passed in by RAGPipeline

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        mode: str = "hybrid",
        source_filter: Optional[str] = None,
        modality_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a natural language query.

        Args:
            query: user's question
            top_k: max results
            mode: "hybrid" | "vector" | "text"
            source_filter: restrict to a specific source filename
            modality_filter: restrict to "text", "table", or "form"

        Returns:
            List of chunk dicts with scores, sorted descending
        """
        top_k = top_k or settings.TOP_K_RESULTS

        logger.info("Retrieving (mode=%s, k=%d): %s", mode, top_k, query[:80])

        query_vector = self.embedder.embed_text(query)

        if mode == "vector":
            results = self.os_client.knn_search(
                query_vector=query_vector,
                k=top_k,
                source_filter=source_filter,
                modality_filter=modality_filter,
            )
        elif mode == "hybrid":
            results = self.os_client.hybrid_search(
                query_text=query,
                query_vector=query_vector,
                k=top_k,
            )
            # Apply post-filters for hybrid (since hybrid doesn't support inline filters)
            if source_filter:
                results = [r for r in results if r["source"] == source_filter]
            if modality_filter:
                results = [r for r in results if r["modality"] == modality_filter]
        else:
            raise ValueError(f"Unknown retrieval mode: {mode}")

        logger.info("Retrieved %d chunks", len(results))
        return results
