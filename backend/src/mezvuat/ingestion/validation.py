"""Upload validation: the bytes decide the type, not the caller's filename."""

from __future__ import annotations

import io
from pathlib import PurePath

from pypdf import PdfReader
from pypdf.errors import PyPdfError

from mezvuat.core.errors import ParseFailedError, PayloadTooLargeError, UnsupportedMediaError

SOURCE_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "markdown": "text/markdown",
    "text": "text/plain",
    "html": "text/html",
}
_MARKDOWN_SUFFIXES = {".md", ".markdown"}
_TEXT_SUFFIXES = {".txt", ".text"}


def detect_source_type(raw: bytes, filename: str | None) -> str:
    """Return `pdf`, `markdown` or `text`, from magic bytes and content; the extension only separates md from txt."""
    if raw.startswith(b"%PDF-"):
        return "pdf"
    suffix = PurePath(filename).suffix.lower() if filename else ""
    if suffix == ".pdf":
        raise UnsupportedMediaError("file is named .pdf but is not a PDF", code="content_type_mismatch")
    if suffix and suffix not in _MARKDOWN_SUFFIXES | _TEXT_SUFFIXES:
        raise UnsupportedMediaError(f"unsupported file type {suffix!r}")
    if b"\x00" in raw:
        raise UnsupportedMediaError("binary content is not supported", code="content_type_mismatch")
    try:
        raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise UnsupportedMediaError("text files must be UTF-8", code="invalid_encoding") from exc
    return "markdown" if suffix in _MARKDOWN_SUFFIXES else "text"


def validate_size(raw: bytes, max_bytes: int) -> None:
    if not raw:
        raise ParseFailedError("file is empty", code="empty_document")
    if len(raw) > max_bytes:
        raise PayloadTooLargeError(f"file exceeds the {max_bytes} byte limit")


def validate_pdf(raw: bytes, max_pages: int) -> None:
    try:
        reader = PdfReader(io.BytesIO(raw), strict=False)
        encrypted = reader.is_encrypted
        pages = 0 if encrypted else len(reader.pages)
    except (PyPdfError, ValueError, KeyError, TypeError, RecursionError, OSError) as exc:
        raise ParseFailedError("PDF could not be read", code="corrupt_pdf") from exc
    if encrypted:
        raise ParseFailedError("encrypted PDFs are not supported", code="encrypted_pdf")
    if pages == 0:
        raise ParseFailedError("PDF has no pages", code="empty_document")
    if pages > max_pages:
        raise PayloadTooLargeError(f"PDF has {pages} pages; the limit is {max_pages}", code="too_many_pages")
