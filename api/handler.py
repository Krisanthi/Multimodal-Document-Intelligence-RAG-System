"""
AWS Lambda Handler — API Gateway integration for the RAG pipeline.

Endpoints:
  POST /ingest   — upload a document (base64-encoded body)
  POST /query    — ask a question
  GET  /status   — system health & stats
  DELETE /source — remove an indexed document
"""

import json
import base64
import logging
from typing import Dict, Any

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Lazy-init pipeline (cold start optimization)
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from pipeline import RAGPipeline
        _pipeline = RAGPipeline(lazy_init=True)
    return _pipeline


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda entry point for API Gateway proxy integration."""
    http_method = event.get("httpMethod", "GET")
    path = event.get("path", "/")
    body = event.get("body", "{}")

    logger.info("Request: %s %s", http_method, path)

    try:
        if isinstance(body, str):
            body = json.loads(body) if body else {}
    except json.JSONDecodeError:
        return _response(400, {"error": "Invalid JSON body"})

    # Route
    if path == "/ingest" and http_method == "POST":
        return _handle_ingest(body)
    elif path == "/query" and http_method == "POST":
        return _handle_query(body)
    elif path == "/status" and http_method == "GET":
        return _handle_status()
    elif path == "/source" and http_method == "DELETE":
        return _handle_delete_source(body)
    else:
        return _response(404, {"error": f"Not found: {http_method} {path}"})


def _handle_ingest(body: Dict) -> Dict:
    """Ingest a base64-encoded document."""
    filename = body.get("filename")
    file_data = body.get("file_data")  # base64

    if not filename or not file_data:
        return _response(400, {"error": "Missing 'filename' or 'file_data'"})

    try:
        file_bytes = base64.b64decode(file_data)
    except Exception as e:
        return _response(400, {"error": f"Invalid base64 data: {str(e)}"})

    pipeline = get_pipeline()
    result = pipeline.ingest(filename, file_bytes)
    return _response(200, result)


def _handle_query(body: Dict) -> Dict:
    """Answer a question using the RAG pipeline."""
    question = body.get("question")
    if not question:
        return _response(400, {"error": "Missing 'question'"})

    pipeline = get_pipeline()
    result = pipeline.query(
        question=question,
        top_k=body.get("top_k"),
        rerank_top_k=body.get("rerank_top_k"),
        search_mode=body.get("search_mode", "hybrid"),
        source_filter=body.get("source_filter"),
    )

    # Remove embedding vectors from response (too large)
    for chunk in result.get("retrieved_chunks", []):
        chunk.pop("embedding", None)

    return _response(200, result)


def _handle_status() -> Dict:
    """Return system status."""
    pipeline = get_pipeline()
    status = pipeline.get_status()
    return _response(200, status)


def _handle_delete_source(body: Dict) -> Dict:
    """Delete all chunks for a source document."""
    source = body.get("source")
    if not source:
        return _response(400, {"error": "Missing 'source'"})

    pipeline = get_pipeline()
    deleted = pipeline.delete_source(source)
    return _response(200, {"source": source, "deleted": deleted})


def _response(status_code: int, body: Dict) -> Dict:
    """Format an API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body, default=str),
    }
