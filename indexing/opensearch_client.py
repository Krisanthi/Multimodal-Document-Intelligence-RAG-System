"""
OpenSearch Client — connection management, indexing, and search operations.
"""

import logging
from typing import List, Dict, Any, Optional

from opensearchpy import OpenSearch, RequestsHttpConnection, helpers

from config.settings import settings

logger = logging.getLogger(__name__)


class OpenSearchClient:
    """Wrapper for OpenSearch operations: index management, bulk insert, KNN search."""

    def __init__(self):
        self.index_name = settings.OPENSEARCH_INDEX
        self.client = self._create_client()

    def _create_client(self) -> OpenSearch:
        """Create and return an OpenSearch client."""
        auth = (settings.OPENSEARCH_USER, settings.OPENSEARCH_PASSWORD)
        host = settings.OPENSEARCH_HOST
        port = settings.OPENSEARCH_PORT
        use_ssl = settings.OPENSEARCH_USE_SSL

        client = OpenSearch(
            hosts=[{"host": host, "port": port}],
            http_auth=auth,
            use_ssl=use_ssl,
            verify_certs=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            timeout=60,
        )
        logger.info("OpenSearch client → %s:%d (ssl=%s)", host, port, use_ssl)
        return client

    # ── Index management ─────────────────────────────────────────────

    def create_index(self, dimension: int = None) -> bool:
        """Create the vector index with KNN mapping."""
        dim = dimension or settings.EMBEDDING_DIMENSION

        if self.client.indices.exists(index=self.index_name):
            logger.info("Index '%s' already exists", self.index_name)
            return False

        body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 256,
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                },
            },
            "mappings": {
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "text": {"type": "text", "analyzer": "standard"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 512, "m": 16},
                        },
                    },
                    "modality": {"type": "keyword"},
                    "source": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": True},
                }
            },
        }

        self.client.indices.create(index=self.index_name, body=body)
        logger.info("Created index '%s' with %d-dim KNN", self.index_name, dim)
        return True

    def delete_index(self) -> bool:
        """Delete the index if it exists."""
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
            logger.info("Deleted index '%s'", self.index_name)
            return True
        return False

    def index_exists(self) -> bool:
        return self.client.indices.exists(index=self.index_name)

    # ── Document operations ──────────────────────────────────────────

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Bulk-index a list of chunk dicts (must include 'embedding' key)."""
        actions = []
        for chunk in chunks:
            doc = {
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": chunk["embedding"],
                "modality": chunk.get("modality", "text"),
                "source": chunk.get("source", "unknown"),
                "chunk_index": chunk.get("chunk_index", 0),
                "metadata": chunk.get("metadata", {}),
            }
            action = {
                "_index": self.index_name,
                "_id": chunk["chunk_id"],
                "_source": doc,
            }
            actions.append(action)

        success, errors = helpers.bulk(self.client, actions, raise_on_error=False)
        if errors:
            logger.warning("Bulk indexing errors: %s", errors)
        logger.info("Indexed %d/%d chunks into '%s'", success, len(chunks), self.index_name)
        return success

    def delete_by_source(self, source: str) -> int:
        """Delete all chunks from a given source document."""
        body = {"query": {"term": {"source": source}}}
        resp = self.client.delete_by_query(index=self.index_name, body=body)
        deleted = resp.get("deleted", 0)
        logger.info("Deleted %d chunks for source '%s'", deleted, source)
        return deleted

    # ── Search ───────────────────────────────────────────────────────

    def knn_search(
        self,
        query_vector: List[float],
        k: int = None,
        source_filter: Optional[str] = None,
        modality_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        KNN vector search with optional filters.

        Returns list of dicts with keys: chunk_id, text, score, modality, source, metadata
        """
        k = k or settings.TOP_K_RESULTS

        # Build filter
        filter_clauses = []
        if source_filter:
            filter_clauses.append({"term": {"source": source_filter}})
        if modality_filter:
            filter_clauses.append({"term": {"modality": modality_filter}})

        knn_query = {
            "knn": {
                "embedding": {
                    "vector": query_vector,
                    "k": k,
                }
            }
        }

        if filter_clauses:
            body = {
                "size": k,
                "query": {
                    "bool": {
                        "must": [knn_query],
                        "filter": filter_clauses,
                    }
                },
            }
        else:
            body = {"size": k, "query": knn_query}

        response = self.client.search(index=self.index_name, body=body)

        results = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append({
                "chunk_id": src["chunk_id"],
                "text": src["text"],
                "score": hit["_score"],
                "modality": src.get("modality", "text"),
                "source": src.get("source", "unknown"),
                "chunk_index": src.get("chunk_index", 0),
                "metadata": src.get("metadata", {}),
            })

        return results

    def hybrid_search(
        self,
        query_text: str,
        query_vector: List[float],
        k: int = None,
        vector_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 text search and KNN vector search.
        Results are merged and re-scored using weighted combination.
        """
        k = k or settings.TOP_K_RESULTS
        text_weight = 1.0 - vector_weight

        # KNN results
        knn_results = self.knn_search(query_vector, k=k * 2)

        # BM25 text search
        text_body = {
            "size": k * 2,
            "query": {
                "match": {
                    "text": {
                        "query": query_text,
                        "fuzziness": "AUTO",
                    }
                }
            },
        }
        text_response = self.client.search(index=self.index_name, body=text_body)
        text_results = []
        for hit in text_response["hits"]["hits"]:
            src = hit["_source"]
            text_results.append({
                "chunk_id": src["chunk_id"],
                "text": src["text"],
                "score": hit["_score"],
                "modality": src.get("modality", "text"),
                "source": src.get("source", "unknown"),
                "chunk_index": src.get("chunk_index", 0),
                "metadata": src.get("metadata", {}),
            })

        # Normalize scores
        knn_max = max((r["score"] for r in knn_results), default=1.0) or 1.0
        text_max = max((r["score"] for r in text_results), default=1.0) or 1.0

        merged: Dict[str, Dict] = {}
        for r in knn_results:
            cid = r["chunk_id"]
            merged[cid] = {**r, "score": (r["score"] / knn_max) * vector_weight}

        for r in text_results:
            cid = r["chunk_id"]
            text_score = (r["score"] / text_max) * text_weight
            if cid in merged:
                merged[cid]["score"] += text_score
            else:
                merged[cid] = {**r, "score": text_score}

        merged_list = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return merged_list[:k]

    # ── Stats ────────────────────────────────────────────────────────

    def get_doc_count(self) -> int:
        if not self.index_exists():
            return 0
        resp = self.client.count(index=self.index_name)
        return resp.get("count", 0)

    def get_sources(self) -> List[str]:
        """Get unique source filenames in the index."""
        if not self.index_exists():
            return []
        body = {
            "size": 0,
            "aggs": {
                "sources": {
                    "terms": {"field": "source", "size": 1000}
                }
            },
        }
        resp = self.client.search(index=self.index_name, body=body)
        buckets = resp.get("aggregations", {}).get("sources", {}).get("buckets", [])
        return [b["key"] for b in buckets]

    def health_check(self) -> Dict[str, Any]:
        """Return cluster health information."""
        try:
            health = self.client.cluster.health()
            return {
                "status": health.get("status", "unknown"),
                "number_of_nodes": health.get("number_of_nodes", 0),
                "active_shards": health.get("active_primary_shards", 0),
            }
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
