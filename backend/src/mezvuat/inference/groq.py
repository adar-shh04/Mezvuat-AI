"""Minimal direct Groq client for offline dataset work (ADR-0011 direct backend, structured calls only).

Decision D1: callers may send only public-corpus text through this client. The key is read from settings and is never
logged. The client paces requests from Groq's rate-limit headers and backs off on 429 and 5xx.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

from mezvuat.core.config import Settings
from mezvuat.core.errors import MezvuatError

log = logging.getLogger(__name__)
_DURATION = re.compile(r"(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")


class InferenceError(MezvuatError):
    code = "inference_failed"
    http_status = 502


class StructuredOutputInvalidError(InferenceError):
    code = "structured_output_invalid"


def parse_reset(value: str | None) -> float:
    """Groq reset headers look like `7.66s`, `1m2.5s` or `250ms`. Unknown formats count as one second."""
    if not value:
        return 0.0
    match = _DURATION.match(value.strip())
    if not match or not any(match.groups()):
        return 1.0
    minutes, seconds, millis = (float(g) if g else 0.0 for g in match.groups())
    return minutes * 60 + seconds + millis / 1000


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries_429: int = 0
    seconds_waiting: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class StructuredResult:
    data: dict[str, Any]
    model: str
    prompt_tokens: int
    completion_tokens: int
    request_id: str | None


@dataclass
class GroqClient:
    api_key: str = field(repr=False)
    model: str
    base_url: str = "https://api.groq.com/openai/v1"
    min_interval_s: float = 2.5
    max_retries: int = 6
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=120.0))
    sleep: Callable[[float], None] = time.sleep
    usage: Usage = field(default_factory=Usage)
    _last_call: float = 0.0
    _pause_until: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @classmethod
    def from_settings(cls, settings: Settings, model: str | None = None) -> GroqClient:
        if settings.groq_api_key is None:
            raise InferenceError("GROQ_API_KEY is not configured", code="provider_not_configured")
        return cls(
            api_key=settings.groq_api_key.get_secret_value(),
            model=model or settings.groq_model_small,
            base_url=settings.groq_base_url,
        )

    def _wait_turn(self) -> None:
        now = time.monotonic()
        wait = max(self._last_call + self.min_interval_s - now, self._pause_until - now, 0.0)
        if wait > 0:
            self.usage.seconds_waiting += wait
            self.sleep(wait)
        self._last_call = time.monotonic()

    def _note_limits(self, response: httpx.Response, next_estimate: int) -> None:
        remaining = response.headers.get("x-ratelimit-remaining-tokens")
        if remaining is not None and remaining.isdigit() and int(remaining) < next_estimate:
            self._pause_until = time.monotonic() + parse_reset(response.headers.get("x-ratelimit-reset-tokens"))
        remaining_requests = response.headers.get("x-ratelimit-remaining-requests")
        if remaining_requests is not None and remaining_requests.isdigit() and int(remaining_requests) == 0:
            raise InferenceError("Groq daily request quota is exhausted", code="quota_exhausted")

    def structured(
        self,
        messages: list[dict[str, str]],
        *,
        schema_name: str,
        schema: dict[str, Any],
        temperature: float,
        seed: int,
        max_tokens: int,
        reasoning_effort: str = "low",
        estimated_prompt_tokens: int = 2000,
    ) -> StructuredResult:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "max_completion_tokens": max_tokens,
            "reasoning_effort": reasoning_effort,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for attempt in range(self.max_retries + 1):
            with self._lock:
                self._wait_turn()
            try:
                response = self.client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            except httpx.TransportError as exc:
                if attempt == self.max_retries:
                    raise InferenceError(f"Groq unreachable: {type(exc).__name__}") from exc
                self.sleep(min(2**attempt, 30))
                continue
            if response.status_code == 429:
                self.usage.retries_429 += 1
                retry_after = response.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.replace(".", "").isdigit() else 2**attempt
                if delay > 600:
                    raise InferenceError("Groq rate limit requires a wait over 10 minutes", code="quota_exhausted")
                self.usage.seconds_waiting += delay
                self.sleep(delay)
                continue
            if response.status_code >= 500 and attempt < self.max_retries:
                self.sleep(min(2**attempt, 30))
                continue
            if response.status_code != 200:
                # The body can echo request content; keep only the provider error type and code.
                error = (
                    response.json().get("error", {})
                    if response.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                raise InferenceError(
                    f"Groq returned {response.status_code} {error.get('type', '')} {error.get('code', '')}".strip()
                )
            payload = response.json()
            usage = payload.get("usage", {})
            self.usage.calls += 1
            self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
            self.usage.completion_tokens += int(usage.get("completion_tokens", 0))
            self._note_limits(response, estimated_prompt_tokens + max_tokens)
            content = payload["choices"][0]["message"].get("content") or ""
            try:
                data = json.loads(content)
            except json.JSONDecodeError as exc:
                raise StructuredOutputInvalidError("model returned invalid JSON despite strict mode") from exc
            return StructuredResult(
                data=data,
                model=str(payload.get("model", self.model)),
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                request_id=response.headers.get("x-request-id"),
            )
        raise InferenceError("Groq request failed after retries", code="retries_exhausted")
