#!/bin/bash
# One-time setup for DocIntel RAG

set -e

echo ""
echo "DocIntel RAG - First-Time Setup"
echo "================================"
echo ""

echo "[1/4] Checking Python..."
if command -v python3 &> /dev/null; then
    echo "  Found $(python3 --version)"
else
    echo "  Python3 not found. Install from https://www.python.org/downloads/"
    exit 1
fi

echo "[2/4] Checking Docker..."
if command -v docker &> /dev/null; then
    if docker info &> /dev/null 2>&1; then
        echo "  Docker is running"
    else
        echo "  Docker installed but not running. Open Docker Desktop first."
        exit 1
    fi
else
    echo "  Docker not found. Install from https://www.docker.com/products/docker-desktop/"
    exit 1
fi

echo "[3/4] Setting up Python environment..."
cd "$(dirname "$0")"
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "  Packages installed"

echo "[4/4] Downloading AI models..."
python3 -c "
import warnings
warnings.filterwarnings('ignore')
from sentence_transformers import SentenceTransformer, CrossEncoder
print('  Loading embedding model...')
SentenceTransformer('all-MiniLM-L6-v2', cache_folder='models_cache')
print('  Loading reranker model...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
print('  Models ready')
"

echo ""
echo "Setup complete. Run ./start.sh to launch."
echo ""
