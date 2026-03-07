#!/usr/bin/env python3
"""
Standalone script to set up the OpenSearch vector index.

Usage:
    python scripts/setup_opensearch.py
    python scripts/setup_opensearch.py --recreate
    python scripts/setup_opensearch.py --status
"""

import sys
import argparse
import logging
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from indexing.opensearch_client import OpenSearchClient

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="OpenSearch Index Management")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the index")
    parser.add_argument("--status", action="store_true", help="Show index status only")
    parser.add_argument("--delete", action="store_true", help="Delete the index")
    args = parser.parse_args()

    client = OpenSearchClient()

    # Health check
    health = client.health_check()
    print(f"\n{'═' * 60}")
    print(f"  OpenSearch Cluster Health")
    print(f"{'═' * 60}")
    print(f"  Status:       {health.get('status', 'unknown')}")
    print(f"  Nodes:        {health.get('number_of_nodes', '?')}")
    print(f"  Host:         {settings.OPENSEARCH_HOST}:{settings.OPENSEARCH_PORT}")
    print(f"  Index:        {settings.OPENSEARCH_INDEX}")
    print(f"{'═' * 60}\n")

    if health.get("status") == "unreachable":
        print("❌ Cannot reach OpenSearch. Is it running?")
        print("   Try: docker-compose up opensearch")
        sys.exit(1)

    if args.status:
        exists = client.index_exists()
        print(f"  Index exists:   {exists}")
        if exists:
            count = client.get_doc_count()
            sources = client.get_sources()
            print(f"  Total chunks:   {count}")
            print(f"  Sources:        {len(sources)}")
            for src in sources:
                print(f"    • {src}")
        sys.exit(0)

    if args.delete:
        if client.delete_index():
            print("✅ Index deleted")
        else:
            print("ℹ️  Index does not exist")
        sys.exit(0)

    if args.recreate and client.index_exists():
        print("⚠️  Deleting existing index...")
        client.delete_index()

    created = client.create_index(dimension=settings.EMBEDDING_DIMENSION)
    if created:
        print(f"✅ Index '{settings.OPENSEARCH_INDEX}' created")
        print(f"   Dimension: {settings.EMBEDDING_DIMENSION}")
        print(f"   Engine:    HNSW (nmslib)")
        print(f"   Space:     cosinesimil")
    else:
        print(f"ℹ️  Index '{settings.OPENSEARCH_INDEX}' already exists")


if __name__ == "__main__":
    main()
