"""HTTP helpers shared by the direct and ModelMesh backends. Error bodies can echo the prompt, so only the status,
error type and error code are ever kept."""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

RETRYABLE_STATUS = frozenset({500, 502, 503, 504})


def retry_after(response: httpx.Response) -> float | None:
    """`Retry-After` in seconds (integer or decimal). HTTP-date values are not used by either backend."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


@dataclass(frozen=True)
class ErrorInfo:
    status: int
    type: str | None
    code: str | None
    request_id: str | None

    @property
    def summary(self) -> str:
        return " ".join(p for p in (str(self.status), self.type or "", self.code or "") if p)


def error_info(response: httpx.Response) -> ErrorInfo:
    """The safe part of an error body: `error.type`, `error.code` and `error.request_id` (contract §3)."""
    error: object = {}
    if "json" in response.headers.get("content-type", ""):
        try:
            body = response.json()
            error = body.get("error", {}) if isinstance(body, dict) else {}
        except (json.JSONDecodeError, UnicodeDecodeError):
            error = {}
    if not isinstance(error, dict):
        error = {}

    def text(key: str) -> str | None:
        value = error.get(key)
        return str(value)[:64] if isinstance(value, str) and value else None

    return ErrorInfo(response.status_code, text("type"), text("code"), text("request_id"))
