"""Cross-encoder reranking of retrieved chunks."""

import logging
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder

from config.settings import settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder based reranker.

    Default model: cross-encoder/ms-marco-MiniLM-L-6-v2
    (fast, ~22 MB, good quality for reranking)
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = None):
        model_name = model_name or self.DEFAULT_MODEL
        cache_dir = str(settings.MODELS_CACHE_DIR)
        logger.info("Loading reranker model: %s", model_name)
        self.model = CrossEncoder(model_name, max_length=512)
        self._model_name = model_name

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = None,
    ) -> List[Dict[str, Any]]:
        """
        Rerank chunks by cross-encoder relevance to the query.

        Args:
            query: the user's question
            chunks: list of chunk dicts from retrieval
            top_k: how many to keep after reranking

        Returns:
            Reranked list (descending score), truncated to top_k
        """
        top_k = top_k or settings.RERANK_TOP_K

        if not chunks:
            return []

        if len(chunks) <= 1:
            return chunks[:top_k]

        # Build query-doc pairs for the cross-encoder
        pairs = [(query, chunk["text"]) for chunk in chunks]

        # Score all pairs
        scores = self.model.predict(pairs)

        # Attach rerank scores to chunks
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Sort by rerank score descending
        reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

        logger.info("Reranked %d to top %d (best=%.4f)",
            len(chunks), min(top_k, len(reranked)), reranked[0]["rerank_score"])

        return reranked[:top_k]
