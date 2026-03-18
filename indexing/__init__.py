"""Indexing package. Provides OpenSearchClient and FAISSClient.

Imports are lazy to avoid loading backends that aren't configured.
Use the vector_client() helper or import the specific class you need.
"""

__all__ = ["OpenSearchClient", "FAISSClient"]


def __getattr__(name):
    """Lazy import: only load the requested backend."""
    if name == "OpenSearchClient":
        from .opensearch_client import OpenSearchClient
        return OpenSearchClient
    if name == "FAISSClient":
        from .faiss_client import FAISSClient
        return FAISSClient
    raise AttributeError(f"module 'indexing' has no attribute {name!r}")