"""
OpenSearch Index Setup Script

Run standalone to create/recreate the vector index:
    python -m indexing.setup_index [--recreate]
"""

import argparse
import logging
import sys

from config.settings import settings
from indexing.opensearch_client import OpenSearchClient

logging.basicConfig(level=settings.LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def setup_opensearch_index(recreate: bool = False) -> bool:
    """Create the OpenSearch index. Optionally delete and recreate."""
    client = OpenSearchClient()

    # Health check
    health = client.health_check()
    logger.info("OpenSearch cluster health: %s", health)

    if health.get("status") == "unreachable":
        logger.error("Cannot reach OpenSearch at %s:%d", settings.OPENSEARCH_HOST, settings.OPENSEARCH_PORT)
        return False

    if recreate and client.index_exists():
        logger.warning("Recreate flag set — deleting existing index '%s'", settings.OPENSEARCH_INDEX)
        client.delete_index()

    created = client.create_index(dimension=settings.EMBEDDING_DIMENSION)
    if created:
        logger.info("✅ Index '%s' created successfully", settings.OPENSEARCH_INDEX)
    else:
        logger.info("ℹ️  Index '%s' already exists", settings.OPENSEARCH_INDEX)

    return True


def main():
    parser = argparse.ArgumentParser(description="Setup OpenSearch vector index")
    parser.add_argument("--recreate", action="store_true", help="Delete and recreate the index")
    args = parser.parse_args()

    success = setup_opensearch_index(recreate=args.recreate)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
