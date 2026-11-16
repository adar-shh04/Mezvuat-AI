"""ModelMesh endpoints other than chat completions (contract §4, §5).

- `POST /v1/feedback/batch`: quality labels, at most 500 per call. Per-item outcomes: `stored` and `replayed` are
  fine; `conflict` (same idempotency key, different score) is logged as an evaluator nondeterminism bug and is not
  overwritten; `not_found` and `invalid` are logged.
- `DELETE /v1/feedback/{request_id}`: data deletion. A 404 means there is nothing left to delete.
- `POST /v1/cache/invalidate {namespace}`: workspace deletion purges the workspace's cache partition.
- `GET /v1/requests/{id}`: the execution record, for cost and latency in the combined benchmark.

429 waits for `Retry-After` (bounded), 5xx and network errors retry with backoff, any other 4xx fails. Labels never
carry content: `notes` is always null.
"""

from __future__ import annotations

import logging
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mezvuat.core.ids import new_id
from mezvuat.inference.errors import InferenceRejectedError, ModelMeshUnavailableError
from mezvuat.inference.http import RETRYABLE_STATUS, error_info, retry_after
from mezvuat.inference.modelmesh.keys import FINGERPRINT, NAMESPACE

log = logging.getLogger(__name__)

MMR_ID = re.compile(r"^mmr_[0-9A-HJKMNP-TV-Z]{26}$")
METRIC = re.compile(r"^[a-z0-9_.:-]{1,64}$")
METRIC_VERSION = re.compile(r"^[A-Za-z0-9_.:+-]{1,64}$")
BATCH_LIMIT = 500


class FeedbackItem(BaseModel):
    """`FeedbackRequest` in openapi-v1.yaml. The idempotency key is (request_id, metric, metric_version, evaluator)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    label_source: Literal["automated_eval", "human_review", "user_signal"]
    metric: str
    metric_version: str
    score: float = Field(ge=0.0, le=1.0)
    higher_is_better: bool = True
    accepted: bool | None = None
    attribution: Literal["direct", "workflow"] = "direct"
    evaluator: str = Field(min_length=1, max_length=128)
    eval_run_id: str | None = Field(default=None, max_length=128)
    dataset_version: str | None = Field(default=None, max_length=128)
    labelled_at: datetime | None = None
    task_type: str | None = Field(default=None, max_length=64)
    notes: None = None

    @field_validator("request_id")
    @classmethod
    def _mmr(cls, value: str) -> str:
        if not MMR_ID.match(value):
            raise ValueError("request_id must be a ModelMesh mmr_ id")
        return value

    @field_validator("metric")
    @classmethod
    def _metric(cls, value: str) -> str:
        if not METRIC.match(value):
            raise ValueError("metric must match ^[a-z0-9_.:-]{1,64}$")
        return value

    @field_validator("metric_version")
    @classmethod
    def _metric_version(cls, value: str) -> str:
        if not METRIC_VERSION.match(value):
            raise ValueError("metric_version must match ^[A-Za-z0-9_.:+-]{1,64}$")
        return value

    def wire(self) -> dict[str, Any]:
        body = self.model_dump(mode="json", exclude_none=True)
        body["notes"] = None
        return body


@dataclass
class FeedbackOutcome:
    statuses: Counter[str] = field(default_factory=Counter)
    conflicts: list[int] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return set(self.statuses) <= {"stored", "replayed"}


def user_thumbs_items(
    request_ids: Sequence[str], *, rating: Literal["up", "down"], labelled_at: datetime
) -> list[FeedbackItem]:
    """A thumbs rating on a whole answer, shared by every ModelMesh request of the run. `user_signal` with
    `attribution: workflow`, so ModelMesh never treats it as training data (contract §5)."""
    return [
        FeedbackItem(
            request_id=request_id,
            label_source="user_signal",
            metric="user_thumbs",
            metric_version="1",
            score=1.0 if rating == "up" else 0.0,
            accepted=rating == "up",
            attribution="workflow",
            evaluator="mezvuat-ui/thumbs",
            labelled_at=labelled_at,
        )
        for request_id in dict.fromkeys(request_ids)
        if MMR_ID.match(request_id)
    ]


@dataclass
class ModelMeshOps:
    base_url: str
    api_key: str | None = field(default=None, repr=False)
    http: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=30.0))
    sleep: Callable[[float], None] = time.sleep
    jitter: Callable[[], float] = random.random
    max_retries: int = 3
    max_wait_s: float = 60.0

    def _headers(self) -> dict[str, str]:
        headers = {"X-Request-ID": new_id("mmops")}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _call(
        self, method: str, path: str, *, json: Any = None, params: dict[str, str] | None = None
    ) -> httpx.Response:
        """One logical call with the retry rules above. Returns any non-retryable response."""
        failures = 0
        while True:
            try:
                response = self.http.request(
                    method, f"{self.base_url}{path}", json=json, params=params, headers=self._headers()
                )
            except httpx.TransportError as exc:
                if failures >= self.max_retries:
                    raise ModelMeshUnavailableError(
                        f"ModelMesh unreachable: {type(exc).__name__}", reason="modelmesh_unreachable"
                    ) from exc
                self.sleep(0.5 * (2**failures) * self.jitter())
                failures += 1
                continue
            if response.status_code == 429 or response.status_code in RETRYABLE_STATUS:
                if failures >= self.max_retries:
                    if response.status_code == 429:
                        return response
                    raise ModelMeshUnavailableError(
                        f"ModelMesh returned {error_info(response).summary}",
                        reason=f"modelmesh_{response.status_code}",
                    )
                hint = retry_after(response)
                wait = hint if hint is not None else 0.5 * (2**failures) * self.jitter()
                if wait > self.max_wait_s:
                    if response.status_code == 429:
                        return response
                    raise ModelMeshUnavailableError(
                        "ModelMesh asked for a longer wait than allowed", reason=f"modelmesh_{response.status_code}"
                    )
                self.sleep(wait)
                failures += 1
                continue
            return response

    @staticmethod
    def _fail(response: httpx.Response, what: str) -> InferenceRejectedError:
        info = error_info(response)
        code = "inference_rate_limited" if info.status == 429 else "inference_rejected"
        return InferenceRejectedError(f"ModelMesh {what}: {info.summary}", code=code)

    def post_feedback(self, items: Sequence[FeedbackItem]) -> FeedbackOutcome:
        outcome = FeedbackOutcome()
        for start in range(0, len(items), BATCH_LIMIT):
            batch = items[start : start + BATCH_LIMIT]
            response = self._call("POST", "/v1/feedback/batch", json={"items": [item.wire() for item in batch]})
            if response.status_code != 200:
                raise self._fail(response, "rejected a feedback batch")
            results = response.json().get("results", [])
            for result in results:
                status = str(result.get("status", "invalid"))
                index = int(result.get("index", -1)) + start
                outcome.statuses[status] += 1
                if status == "conflict":
                    outcome.conflicts.append(index)
                    log.error(
                        "feedback conflict: same key, different score (evaluator nondeterminism); not overwritten",
                        extra={"index": index, "metric": items[index].metric if 0 <= index < len(items) else None},
                    )
                elif status in ("not_found", "invalid"):
                    log.warning("feedback item %s", status, extra={"index": index})
        return outcome

    def delete_feedback(self, request_id: str) -> int:
        if not MMR_ID.match(request_id):
            raise ValueError("not a ModelMesh request id")
        response = self._call("DELETE", f"/v1/feedback/{request_id}")
        if response.status_code == 404:
            return 0
        if response.status_code != 200:
            raise self._fail(response, "refused to delete feedback")
        return int(response.json().get("deleted", 0))

    def invalidate_cache(self, *, namespace: str | None = None, context_fingerprint: str | None = None) -> int:
        body: dict[str, str] = {}
        if namespace is not None:
            if not NAMESPACE.match(namespace):
                raise ValueError("invalid cache namespace")
            body["namespace"] = namespace
        if context_fingerprint is not None:
            if not FINGERPRINT.match(context_fingerprint):
                raise ValueError("invalid context fingerprint")
            body["context_fingerprint"] = context_fingerprint
        if not body:
            raise ValueError("cache invalidation needs a namespace or a fingerprint")
        response = self._call("POST", "/v1/cache/invalidate", json=body)
        if response.status_code != 200:
            raise self._fail(response, "refused the cache invalidation")
        return int(response.json().get("deleted_entries", 0))

    def request_record(self, request_id: str) -> dict[str, Any] | None:
        """The execution record, or None when ModelMesh does not know the id (404)."""
        if not MMR_ID.match(request_id):
            raise ValueError("not a ModelMesh request id")
        response = self._call("GET", f"/v1/requests/{request_id}")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise self._fail(response, "refused the request record")
        record: dict[str, Any] = response.json()
        return record

    def health(self) -> bool:
        """`GET /v1/health` (unauthenticated): 200 means ok or degraded, 503 means down."""
        try:
            response = self.http.get(f"{self.base_url}/v1/health", timeout=5.0)
        except httpx.TransportError:
            return False
        return response.status_code == 200
