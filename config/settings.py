"""Configuration loaded from environment variables.

On Streamlit Cloud, secrets from the app settings are injected into
os.environ so that the rest of the code can read them with os.getenv().
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Streamlit Cloud secrets integration ───────────────────────────────
# Streamlit stores secrets in st.secrets (not in env vars).  Inject them
# into os.environ so every os.getenv() call picks them up automatically.
try:
    import streamlit as st
    for key, value in st.secrets.items():
        if isinstance(value, str):
            os.environ.setdefault(key, value)
except Exception:
    pass  # Not running on Streamlit Cloud — .env / real env vars are used


class Settings:
    """Central configuration."""

    # AWS
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "multimodal-rag-documents")

    # Groq
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # Vector store backend: "opensearch" or "faiss"
    VECTOR_BACKEND: str = os.getenv("VECTOR_BACKEND", "opensearch")

    # OpenSearch (used when VECTOR_BACKEND=opensearch)
    OPENSEARCH_HOST: str = os.getenv("OPENSEARCH_HOST", "localhost")
    OPENSEARCH_PORT: int = int(os.getenv("OPENSEARCH_PORT", "9200"))
    OPENSEARCH_USER: str = os.getenv("OPENSEARCH_USER", "admin")
    OPENSEARCH_PASSWORD: str = os.getenv("OPENSEARCH_PASSWORD", "admin")
    OPENSEARCH_INDEX: str = os.getenv("OPENSEARCH_INDEX", "multimodal-rag-index")
    OPENSEARCH_USE_SSL: bool = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"

    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    # Chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))
    MAX_CHUNKS_PER_DOC: int = int(os.getenv("MAX_CHUNKS_PER_DOC", "500"))

    # Retrieval
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "10"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))

    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MODELS_CACHE_DIR: Path = BASE_DIR / "models_cache"

    # App
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8501"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def ensure_dirs(cls):
        cls.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_aws_credentials(cls) -> dict:
        return {
            "aws_access_key_id": cls.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": cls.AWS_SECRET_ACCESS_KEY,
            "region_name": cls.AWS_REGION,
        }


settings = Settings()
settings.ensure_dirs()
