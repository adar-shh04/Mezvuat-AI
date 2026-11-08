"""Typed domain errors. `code` is a stable machine-readable string surfaced by the API."""

from __future__ import annotations


class MezvuatError(Exception):
    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class NotFoundError(MezvuatError):
    code = "not_found"
    http_status = 404


class ValidationFailedError(MezvuatError):
    """The input is permanently unacceptable; retrying cannot help."""

    code = "invalid_input"
    http_status = 400


class UnsupportedMediaError(ValidationFailedError):
    code = "unsupported_media_type"
    http_status = 415


class PayloadTooLargeError(ValidationFailedError):
    code = "payload_too_large"
    http_status = 413


class UrlRejectedError(ValidationFailedError):
    code = "url_rejected"


class ParseFailedError(ValidationFailedError):
    code = "parse_failed"
    http_status = 422


class RetrievalUnavailableError(MezvuatError):
    """A retrieval backend is down. The caller must report this; it must never answer without evidence."""

    code = "retrieval_unavailable"
    http_status = 503


class IndexNotReadyError(MezvuatError):
    code = "index_not_ready"
    http_status = 409
