"""Document parser using local OCR libraries with AWS S3 temporary storage.

Parsing pipeline:
  1. Upload file to S3 temporarily (demonstrates AWS integration)
  2. Parse locally with pdfplumber (PDFs) or pytesseract + Pillow (images)
  3. Delete file from S3 after processing

This avoids AWS Textract costs while still showcasing S3 usage.
"""

import io
import logging
from pathlib import Path
from typing import List, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parses documents using pdfplumber (PDFs) and pytesseract (images).
    Uses AWS S3 for temporary file staging to demonstrate cloud integration."""

    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    SUPPORTED_DOC_EXTS = {".pdf"} | SUPPORTED_IMAGE_EXTS

    def __init__(self):
        creds = settings.get_aws_credentials()
        try:
            self._s3 = boto3.client("s3", **creds)
            self._bucket = settings.S3_BUCKET_NAME
            self._s3_available = True
        except Exception as e:
            logger.warning("S3 not available: %s. Continuing without S3.", e)
            self._s3 = None
            self._bucket = None
            self._s3_available = False

    def parse(self, file_path: str, file_bytes: Optional[bytes] = None) -> Dict:
        """Parse a document and return structured content.

        Args:
            file_path: original filename (or path)
            file_bytes: raw file content

        Returns:
            Dict with text_blocks, tables, key_value_pairs, metadata
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_DOC_EXTS:
            raise ValueError(f"Unsupported type: {ext}")
        if file_bytes is None:
            file_bytes = path.read_bytes()

        logger.info("Parsing %s (%.1f KB)", path.name, len(file_bytes) / 1024)

        # Stage 1: Temporarily store in S3 (demonstrates AWS S3 integration)
        s3_key = self._upload_to_s3(path.name, file_bytes)

        # Stage 2: Parse locally (zero cost)
        try:
            if ext == ".pdf":
                result = self._parse_pdf(file_bytes)
            elif ext in self.SUPPORTED_IMAGE_EXTS:
                result = self._parse_image(file_bytes, ext)
            else:
                result = {"text_blocks": [], "tables": [], "key_value_pairs": [],
                          "metadata": {}}
        finally:
            # Stage 3: Clean up S3 temporary file
            if s3_key:
                self._delete_from_s3(s3_key)

        result["source"] = path.name
        result["metadata"]["file_type"] = ext
        result["metadata"]["file_size_bytes"] = len(file_bytes)
        logger.info("Parsed %s: %d text blocks, %d tables",
                     path.name, len(result["text_blocks"]), len(result["tables"]))
        return result

    # ── S3 temporary storage ──────────────────────────────────────────

    def _upload_to_s3(self, filename: str, file_bytes: bytes) -> Optional[str]:
        """Upload file to S3 temporarily. Returns the S3 key or None."""
        if not self._s3_available:
            return None
        s3_key = f"uploads/{filename}"
        try:
            if not self._ensure_bucket():
                return None
            self._s3.put_object(Bucket=self._bucket, Key=s3_key, Body=file_bytes)
            logger.info("Staged %s in S3 (%s/%s)", filename, self._bucket, s3_key)
            return s3_key
        except Exception as e:
            logger.warning("S3 upload failed (continuing without S3): %s", e)
            return None

    def _delete_from_s3(self, s3_key: str):
        """Delete the temporary file from S3."""
        try:
            self._s3.delete_object(Bucket=self._bucket, Key=s3_key)
            logger.info("Cleaned up S3: %s/%s", self._bucket, s3_key)
        except Exception as e:
            logger.warning("S3 cleanup failed: %s", e)

    def _ensure_bucket(self) -> bool:
        """Make sure the S3 bucket exists, create if not."""
        try:
            self._s3.head_bucket(Bucket=self._bucket)
            return True
        except ClientError as e:
            code = int(e.response["Error"]["Code"])
            if code == 404:
                try:
                    cfg = {}
                    if settings.AWS_REGION != "us-east-1":
                        cfg["CreateBucketConfiguration"] = {
                            "LocationConstraint": settings.AWS_REGION}
                    self._s3.create_bucket(Bucket=self._bucket, **cfg)
                    return True
                except Exception as ce:
                    logger.warning("Cannot create bucket: %s", ce)
            return False

    # ── PDF parsing (pdfplumber) ──────────────────────────────────────

    def _parse_pdf(self, file_bytes: bytes) -> Dict:
        """Extract text and tables from PDF using pdfplumber."""
        import pdfplumber

        text_blocks, tables = [], []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            npages = len(pdf.pages)
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    for line in text.split("\n"):
                        line = line.strip()
                        if line:
                            text_blocks.append(line)
                for tbl in (page.extract_tables() or []):
                    md = self._list_table_md(tbl)
                    if md:
                        tables.append(md)

        return {
            "text_blocks": text_blocks,
            "tables": tables,
            "key_value_pairs": [],
            "metadata": {
                "num_pages": npages,
                "num_text_blocks": len(text_blocks),
                "num_tables": len(tables),
                "parser": "pdfplumber",
            },
        }

    # ── Image parsing (pytesseract + Pillow) ──────────────────────────

    def _parse_image(self, file_bytes: bytes, ext: str) -> Dict:
        """Extract text from image using Tesseract OCR via pytesseract."""
        from PIL import Image
        import pytesseract

        image = Image.open(io.BytesIO(file_bytes))

        # Run OCR
        ocr_text = pytesseract.image_to_string(image)

        text_blocks = []
        for line in ocr_text.split("\n"):
            line = line.strip()
            if line:
                text_blocks.append(line)

        # Attempt table detection via pytesseract TSV output
        tables = []
        try:
            tsv_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            tables = self._tsv_to_tables(tsv_data)
        except Exception as e:
            logger.debug("Table detection skipped: %s", e)

        return {
            "text_blocks": text_blocks,
            "tables": tables,
            "key_value_pairs": [],
            "metadata": {
                "num_pages": 1,
                "num_text_blocks": len(text_blocks),
                "num_tables": len(tables),
                "parser": "pytesseract",
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _tsv_to_tables(tsv_data: Dict) -> List[str]:
        """Convert pytesseract TSV output into markdown tables if blocks look tabular."""
        blocks = {}
        for i, block_num in enumerate(tsv_data.get("block_num", [])):
            text = tsv_data["text"][i].strip() if tsv_data["text"][i] else ""
            if not text:
                continue
            line_num = tsv_data["line_num"][i]
            key = (block_num, line_num)
            blocks.setdefault(key, []).append(text)

        # Group into lines — only create a "table" if we have at least 2 rows
        # with multiple columns (heuristic: lines with 3+ words each)
        lines = []
        for key in sorted(blocks):
            row = blocks[key]
            lines.append(row)

        if len(lines) < 2:
            return []

        # Check if it looks tabular (most rows have similar word counts)
        word_counts = [len(row) for row in lines]
        avg_wc = sum(word_counts) / len(word_counts)
        if avg_wc < 2:
            return []

        # Build markdown table
        max_cols = max(len(row) for row in lines)
        md_lines = []
        for i, row in enumerate(lines):
            while len(row) < max_cols:
                row.append("")
            md_lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")

        return ["\n".join(md_lines)] if md_lines else []

    @staticmethod
    def _list_table_md(table: list) -> str:
        """Convert a pdfplumber table (list of lists) to markdown."""
        if not table or not table[0]:
            return ""
        cleaned = [[str(c).strip() if c else "" for c in row] for row in table]
        nc = max(len(r) for r in cleaned)
        lines = []
        for i, row in enumerate(cleaned):
            while len(row) < nc:
                row.append("")
            lines.append("| " + " | ".join(row) + " |")
            if i == 0:
                lines.append("| " + " | ".join(["---"] * nc) + " |")
        return "\n".join(lines)
