"""FAISS-based vector store. Replaces OpenSearch for cloud/serverless deployment.
No Docker or external database needed. Stores index in memory with pickle persistence."""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

logger = logging.getLogger(__name__)

INDEX_DIR = Path("data/faiss_index")


class FAISSClient:
    """In-process vector store using FAISS. Drop-in replacement for OpenSearchClient."""

    def __init__(self, dimension: int = 384, index_path: str = None):
        self._dim = dimension
        self._index_dir = Path(index_path) if index_path else INDEX_DIR
        self._index: Optional[faiss.IndexFlatIP] = None
        self._chunks: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        idx_file = self._index_dir / "index.faiss"
        meta_file = self._index_dir / "chunks.pkl"
        if idx_file.exists() and meta_file.exists():
            self._index = faiss.read_index(str(idx_file))
            with open(meta_file, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info("Loaded FAISS index: %d vectors", self._index.ntotal)
        else:
            self._index = faiss.IndexFlatIP(self._dim)
            self._chunks = []

    def _save(self):
        self._index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_dir / "index.faiss"))
        with open(self._index_dir / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)

    def create_index(self, dimension: int = None):
        if dimension and dimension != self._dim:
            self._dim = dimension
            self._index = faiss.IndexFlatIP(self._dim)
            self._chunks = []
            self._save()

    def health_check(self) -> Dict:
        return {"status": "green", "number_of_nodes": 1, "backend": "faiss"}

    def get_doc_count(self) -> int:
        return len(self._chunks)

    def get_sources(self) -> List[str]:
        return sorted(set(c.get("source", "") for c in self._chunks))

    def index_chunks(self, chunks: List[Dict]) -> int:
        if not chunks:
            return 0
        vecs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        faiss.normalize_L2(vecs)
        self._index.add(vecs)
        for c in chunks:
            meta = {k: v for k, v in c.items() if k != "embedding"}
            self._chunks.append(meta)
        self._save()
        logger.info("Indexed %d chunks (total: %d)", len(chunks), len(self._chunks))
        return len(chunks)

    def knn_search(self, query_vector: List[float], k: int = 10,
                   source_filter: str = None, modality_filter: str = None) -> List[Dict]:
        if self._index.ntotal == 0:
            return []
        vec = np.array([query_vector], dtype=np.float32)
        faiss.normalize_L2(vec)
        scores, indices = self._index.search(vec, min(k * 3, self._index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            chunk = dict(self._chunks[idx])
            if source_filter and chunk.get("source") != source_filter:
                continue
            if modality_filter and chunk.get("modality") != modality_filter:
                continue
            chunk["score"] = float(score)
            results.append(chunk)
            if len(results) >= k:
                break
        return results

    def hybrid_search(self, query_text: str, query_vector: List[float], k: int = 10) -> List[Dict]:
        vec_results = self.knn_search(query_vector, k=k * 2)
        query_terms = set(query_text.lower().split())
        for r in vec_results:
            text_lower = r.get("text", "").lower()
            keyword_hits = sum(1 for t in query_terms if t in text_lower)
            keyword_boost = keyword_hits / max(len(query_terms), 1) * 0.3
            r["score"] = r.get("score", 0) + keyword_boost
        vec_results.sort(key=lambda x: x["score"], reverse=True)
        return vec_results[:k]

    def delete_by_source(self, source: str) -> int:
        keep_idx = [i for i, c in enumerate(self._chunks) if c.get("source") != source]
        removed = len(self._chunks) - len(keep_idx)
        if removed > 0:
            self._chunks = [self._chunks[i] for i in keep_idx]
            self._rebuild_index()
            self._save()
        return removed

    def delete_index(self):
        self._index = faiss.IndexFlatIP(self._dim)
        self._chunks = []
        self._save()

    def _rebuild_index(self):
        self._index = faiss.IndexFlatIP(self._dim)
        if not self._chunks:
            return
        all_vecs = []
        valid_chunks = []
        for c in self._chunks:
            if "embedding" in c:
                all_vecs.append(c["embedding"])
                valid_chunks.append(c)
        if all_vecs:
            vecs = np.array(all_vecs, dtype=np.float32)
            faiss.normalize_L2(vecs)
            self._index.add(vecs)
        self._chunks = valid_chunks
