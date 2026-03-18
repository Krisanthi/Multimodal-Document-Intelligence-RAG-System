from .document_parser import DocumentParser
from .chunker import DocumentChunker
from .embedder import Embedder

__all__ = ["DocumentParser", "DocumentChunker", "Embedder"]


def __init__(self):
    """Initialize the parser with cross-platform OCR and AWS S3 settings."""
    import os
    import pytesseract
    import boto3
    from botocore.exceptions import ClientError

    # 1. Handle Tesseract Path (Cross-Platform Logic)
    # Your Mac uses /opt/homebrew, Streamlit Cloud (Linux) uses /usr/bin
    mac_path = "/opt/homebrew/bin/tesseract"
    linux_path = "/usr/bin/tesseract"

    if os.path.exists(mac_path):
        pytesseract.pytesseract.tesseract_cmd = mac_path
        logger.info("Using local Mac Tesseract path.")
    elif os.path.exists(linux_path):
        pytesseract.pytesseract.tesseract_cmd = linux_path
        logger.info("Using Linux system Tesseract path.")
    else:
        # Fallback: hope it's in the system PATH (Standard for many Linux distros)
        logger.warning("Tesseract path not explicitly found. Using default system PATH.")

    # 2. Handle AWS S3 Setup
    creds = settings.get_aws_credentials()
    try:
        self._s3 = boto3.client("s3", **creds)
        self._bucket = settings.S3_BUCKET_NAME
        self._s3_available = True
        logger.info("S3 client initialized successfully.")
    except Exception as e:
        logger.warning("S3 not available: %s. Continuing with local-only mode.", e)
        self._s3 = None
        self._bucket = None
        self._s3_available = False