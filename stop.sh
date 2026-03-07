#!/bin/bash
# Stop DocIntel RAG

cd "$(dirname "$0")"

echo "Stopping DocIntel RAG..."

if command -v docker &> /dev/null; then
    if docker ps --format '{{.Names}}' | grep -q 'rag-opensearch'; then
        docker compose down 2>/dev/null
        echo "  OpenSearch stopped"
    fi
fi

if pgrep -f "streamlit run" > /dev/null; then
    pkill -f "streamlit run"
    echo "  Streamlit stopped"
fi

echo "Done."
