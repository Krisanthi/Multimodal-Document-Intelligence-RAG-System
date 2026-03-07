#!/bin/bash
# Start DocIntel RAG (OpenSearch + Streamlit)

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Run ./setup.sh first."
    exit 1
fi
source venv/bin/activate

echo ""
echo "DocIntel RAG - Starting"
echo "========================"

# Start OpenSearch
echo "[1/3] OpenSearch..."
if ! docker info &> /dev/null 2>&1; then
    echo "  Docker not running. Open Docker Desktop first."
    exit 1
fi

if docker ps --format '{{.Names}}' | grep -q 'rag-opensearch'; then
    echo "  Already running"
else
    docker compose up opensearch -d 2>&1 | grep -v "WARN"
    for i in {1..45}; do
        if curl -s 'http://localhost:9200/_cluster/health' 2>/dev/null | grep -q '"status"'; then
            echo "  Ready"
            break
        fi
        sleep 2
    done
fi

# Index setup
echo "[2/3] Index..."
python3 scripts/setup_opensearch.py 2>/dev/null && echo "  Ready" || echo "  Will retry on startup"

# Streamlit
echo "[3/3] Launching web app..."
lsof -ti:8501 2>/dev/null | xargs kill -9 2>/dev/null
sleep 1

echo ""
echo "Open http://localhost:8501 in your browser"
echo "Press Ctrl+C to stop"
echo ""

streamlit run ui/app.py --server.port 8501 --server.address localhost --browser.gatherUsageStats false
