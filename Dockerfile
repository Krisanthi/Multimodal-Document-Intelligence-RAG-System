# ── Stage 1: Base ────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies (including Tesseract OCR for image parsing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl tesseract-ocr tesseract-ocr-eng && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 2: App ─────────────────────────────────────────────────────
FROM base AS app

WORKDIR /app
COPY . .

# Create required directories
RUN mkdir -p uploads models_cache

# Download embedding model at build time (faster cold starts)
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2', cache_folder='models_cache')"

# Download reranker model
RUN python -c "from sentence_transformers import CrossEncoder; \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "ui/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
