"""Document parser using AWS Textract with local pdfplumber fallback."""

import io
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional

import boto3
from botocore.exceptions import ClientError
from config.settings import settings

logger = logging.getLogger(__name__)


class DocumentParser:
    """Parses documents via AWS Textract. Falls back to pdfplumber for PDFs
    if Textract is unavailable."""

    SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}
    SUPPORTED_DOC_EXTS = {".pdf"} | SUPPORTED_IMAGE_EXTS

    def __init__(self):
        creds = settings.get_aws_credentials()
        self._textract = boto3.client("textract", **creds)
        self._s3 = boto3.client("s3", **creds)
        self._bucket = settings.S3_BUCKET_NAME
        self._bucket_ok = None

    def parse(self, file_path: str, file_bytes: Optional[bytes] = None) -> Dict:
        path = Path(file_path)
        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_DOC_EXTS:
            raise ValueError(f"Unsupported type: {ext}")
        if file_bytes is None:
            file_bytes = path.read_bytes()

        logger.info("Parsing %s (%.1f KB)", path.name, len(file_bytes) / 1024)

        result = None
        if ext in self.SUPPORTED_IMAGE_EXTS:
            result = self._textract_sync(file_bytes)
        elif ext == ".pdf":
            result = self._textract_pdf(path.name, file_bytes)
            if result is None:
                result = self._local_pdf(file_bytes)

        if result is None:
            result = {"text_blocks": [], "tables": [], "key_value_pairs": [], "metadata": {}}

        result["source"] = path.name
        result["metadata"]["file_type"] = ext
        result["metadata"]["file_size_bytes"] = len(file_bytes)
        logger.info("Parsed %s: %d text blocks, %d tables",
                     path.name, len(result["text_blocks"]), len(result["tables"]))
        return result

    # -- Textract sync (images) --
    def _textract_sync(self, file_bytes: bytes) -> Optional[Dict]:
        try:
            resp = self._textract.analyze_document(
                Document={"Bytes": file_bytes}, FeatureTypes=["TABLES", "FORMS"])
            return self._blocks_to_content(resp.get("Blocks", []))
        except Exception as e:
            logger.warning("Textract sync failed: %s", e)
            return None

    # -- Textract async (PDFs via S3) --
    def _textract_pdf(self, filename: str, file_bytes: bytes) -> Optional[Dict]:
        if not self._ensure_bucket():
            return None
        s3_key = f"uploads/{filename}"
        try:
            self._s3.put_object(Bucket=self._bucket, Key=s3_key, Body=file_bytes)
            resp = self._textract.start_document_analysis(
                DocumentLocation={"S3Object": {"Bucket": self._bucket, "Name": s3_key}},
                FeatureTypes=["TABLES", "FORMS"])
            blocks = self._poll(resp["JobId"])
            return self._blocks_to_content(blocks)
        except Exception as e:
            logger.warning("Textract PDF failed: %s", e)
            return None
        finally:
            try:
                self._s3.delete_object(Bucket=self._bucket, Key=s3_key)
            except Exception:
                pass

    def _poll(self, job_id: str, timeout: int = 300) -> List[Dict]:
        elapsed = 0
        while elapsed < timeout:
            resp = self._textract.get_document_analysis(JobId=job_id)
            if resp["JobStatus"] == "SUCCEEDED":
                blocks = resp.get("Blocks", [])
                token = resp.get("NextToken")
                while token:
                    resp = self._textract.get_document_analysis(JobId=job_id, NextToken=token)
                    blocks.extend(resp.get("Blocks", []))
                    token = resp.get("NextToken")
                return blocks
            if resp["JobStatus"] == "FAILED":
                raise RuntimeError(resp.get("StatusMessage", "Textract job failed"))
            time.sleep(5)
            elapsed += 5
        raise TimeoutError("Textract timed out")

    def _ensure_bucket(self) -> bool:
        if self._bucket_ok is True:
            return True
        try:
            self._s3.head_bucket(Bucket=self._bucket)
            self._bucket_ok = True
            return True
        except ClientError as e:
            code = int(e.response["Error"]["Code"])
            if code == 404:
                try:
                    cfg = {}
                    if settings.AWS_REGION != "us-east-1":
                        cfg["CreateBucketConfiguration"] = {"LocationConstraint": settings.AWS_REGION}
                    self._s3.create_bucket(Bucket=self._bucket, **cfg)
                    self._bucket_ok = True
                    return True
                except Exception as ce:
                    logger.warning("Cannot create bucket: %s", ce)
            self._bucket_ok = False
            return False

    # -- Content extraction from Textract blocks --
    def _blocks_to_content(self, blocks: List[Dict]) -> Dict:
        bmap = {b["Id"]: b for b in blocks}
        text_blocks, tables, kv_pairs = [], [], []
        for b in blocks:
            bt = b.get("BlockType")
            if bt == "LINE" and b.get("Text", "").strip():
                text_blocks.append(b["Text"].strip())
            elif bt == "TABLE":
                md = self._table_md(b, bmap)
                if md:
                    tables.append(md)
            elif bt == "KEY_VALUE_SET" and "KEY" in b.get("EntityTypes", []):
                kv = self._kv_pair(b, bmap)
                if kv:
                    kv_pairs.append(kv)
        pages = {bl.get("Page", 1) for bl in blocks if bl.get("BlockType") == "PAGE"}
        return {"text_blocks": text_blocks, "tables": tables, "key_value_pairs": kv_pairs,
                "metadata": {"num_pages": len(pages) or 1, "num_text_blocks": len(text_blocks),
                             "num_tables": len(tables), "parser": "textract"}}

    def _table_md(self, tbl: Dict, bmap: Dict) -> str:
        rows = {}
        mx = 0
        for rel in tbl.get("Relationships", []):
            if rel["Type"] != "CHILD":
                continue
            for cid in rel["Ids"]:
                c = bmap.get(cid, {})
                if c.get("BlockType") != "CELL":
                    continue
                r, col = c.get("RowIndex", 1), c.get("ColumnIndex", 1)
                mx = max(mx, col)
                rows.setdefault(r, {})[col] = self._text(c, bmap)
        if not rows:
            return ""
        lines = []
        for r in sorted(rows):
            cells = [rows[r].get(i, "") for i in range(1, mx + 1)]
            lines.append("| " + " | ".join(cells) + " |")
            if r == min(rows):
                lines.append("| " + " | ".join(["---"] * mx) + " |")
        return "\n".join(lines)

    def _kv_pair(self, key_block: Dict, bmap: Dict) -> Optional[Dict]:
        k = self._text(key_block, bmap)
        v = ""
        for rel in key_block.get("Relationships", []):
            if rel["Type"] == "VALUE":
                for vid in rel["Ids"]:
                    v = self._text(bmap.get(vid, {}), bmap)
                    break
        return {"key": k.strip(), "value": v.strip()} if k.strip() else None

    def _text(self, block: Dict, bmap: Dict) -> str:
        if "Text" in block:
            return block["Text"]
        parts = []
        for rel in block.get("Relationships", []):
            if rel["Type"] == "CHILD":
                for cid in rel["Ids"]:
                    ch = bmap.get(cid, {})
                    if ch.get("BlockType") == "WORD":
                        parts.append(ch.get("Text", ""))
        return " ".join(parts)

    # -- Local fallback (pdfplumber) --
    def _local_pdf(self, file_bytes: bytes) -> Dict:
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
        return {"text_blocks": text_blocks, "tables": tables, "key_value_pairs": [],
                "metadata": {"num_pages": npages, "num_text_blocks": len(text_blocks),
                             "num_tables": len(tables), "parser": "pdfplumber"}}

    @staticmethod
    def _list_table_md(table: list) -> str:
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
