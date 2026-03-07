"""Embedding module using sentence-transformers (all-MiniLM-L6-v2)."""

import warnings
import logging
from typing import List, Dict

import numpy as np
from config.settings import settings

warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


class Embedder:
    """Converts text into 384-dimensional dense vectors for similarity search."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._model = None
        self._dimension = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(
                self.model_name, cache_folder="models_cache"
            )
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info("Embedding dimension: %d", self._dimension)
        return self._model

    @property
    def embedding_dimension(self) -> int:
        if self._dimension is None:
            _ = self.model
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        vec = self.model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_texts(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        vecs = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True)
        return vecs.tolist()

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        texts = [c["text"] for c in chunks]
        vecs = self.embed_texts(texts)
        for chunk, vec in zip(chunks, vecs):
            chunk["embedding"] = vec
        logger.info("Embedded %d chunks", len(chunks))
        return chunks
