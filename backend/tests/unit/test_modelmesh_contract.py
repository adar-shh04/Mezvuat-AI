"""Contract tests generated from `docs/contract/openapi-v1.yaml` (M8, ADR-0011 verification plan).

They tie the client's wire models and request builders to the frozen spec, so a change on either side fails here:
- the client's parse models have exactly the spec's required fields;
- every extension the request builder can emit is a property the spec defines;
- the spec's own examples parse with the client's models;
- feedback, deletion and cache-purge requests validate against the spec, and their responses are handled per §5.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
import pytest
import respx

from mezvuat.inference import InferenceRejectedError, ModelMeshUnavailableError
from mezvuat.inference.modelmesh import (
    HARD_EXTENSIONS,
    ContractInfo,
    FeedbackItem,
    ModelMeshOps,
    RoutingCalibration,
    user_thumbs_items,
)
from mezvuat.inference.modelmesh.ops import BATCH_LIMIT
from tests.modelmesh_stub import BASE, assert_valid, error, example, mmr_id, spec
from tests.unit.test_modelmesh_client import FakeTime, mesh, request


def schema(name: str) -> dict[str, Any]:
    found: dict[str, Any] = spec()["components"]["schemas"][name]
    return found


# ----- wire models match the spec -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model", "component"), [(ContractInfo, "ContractInfo"), (RoutingCalibration, "RoutingCalibration")]
)
def test_parse_models_require_what_the_spec_requires(model: type, component: str) -> None:
    required = set(schema(component)["required"])
    model_required = {name for name, f in model.model_fields.items() if f.is_required()}
    assert model_required == required


def test_feedback_item_fields_are_the_spec_fields() -> None:
    spec_fields = schema("FeedbackRequest")["properties"]
    assert set(FeedbackItem.model_fields) == set(spec_fields)
    required = set(schema("FeedbackRequest")["required"])
    # ADR-0011: Mezvuat always sets metric_version (it is part of the idempotency key); the rest follows the spec.
    assert {n for n, f in FeedbackItem.model_fields.items() if f.is_required()} == required | {"metric_version"}
    assert FeedbackItem.model_config.get("extra") == "forbid"  # the spec has additionalProperties: false


def test_spec_examples_parse() -> None:
    contract = ContractInfo.model_validate(example("/v1/contract"))
    for path in HARD_EXTENSIONS:
        assert contract.supports(path)
    calibration = RoutingCalibration.model_validate(example("/v1/routing/calibration"))
    assert all(not t.calibrated for t in calibration.task_types)


def test_every_extension_the_builder_emits_exists_in_the_spec() -> None:
    fake = FakeTime()
    client = mesh(fake, min_quality={"evidence_synthesis": 0.7})
    # A contract that lists everything the spec defines, and calibration for the task: the builder emits its maximum.
    every = [f"routing.{k}" for k in schema("RoutingOptions")["properties"]]
    every += [f"cache.{k}" for k in schema("CacheOptions")["properties"]]
    contract = ContractInfo.model_validate({**example("/v1/contract"), "extensions": every})
    body = client.backend.body(
        request(task_type="evidence_synthesis", model="groq/openai/gpt-oss-120b"),
        [{"role": "user", "content": "q"}],
        contract,
        frozenset({"evidence_synthesis"}),
        remaining_s=12.0,
        stream=True,
        strict=True,
    )
    assert set(body["routing"]) <= set(schema("RoutingOptions")["properties"])
    assert set(body["cache"]) <= set(schema("CacheOptions")["properties"])
    assert set(body) <= set(schema("ChatCompletionRequest")["properties"])
    assert_valid("ChatCompletionRequest", body)
    assert body["routing"]["min_quality"] == 0.7
    assert body["routing"]["timeout_ms"] == 12_000


def test_error_statuses_of_the_spec_have_known_types() -> None:
    """Every error type in the spec's enum maps to a status the client's fallback table classifies."""
    types = set(schema("ErrorObject")["properties"]["type"]["enum"])
    from tests.modelmesh_stub import ERROR_TYPES

    assert types == set(ERROR_TYPES.values())


# ----- feedback, deletion, cache purge (contract §4, §5) --------------------------------------------------------------


@pytest.fixture
def router() -> Iterator[respx.Router]:
    with respx.mock(assert_all_called=False) as mocked:
        yield mocked


def ops(fake: FakeTime) -> ModelMeshOps:
    return ModelMeshOps(base_url=BASE, api_key="mm_test_key", http=httpx.Client(), sleep=fake.sleep, jitter=lambda: 0.5)


def label(request_id: str, score: float = 0.8, **overrides: Any) -> FeedbackItem:
    values: dict[str, Any] = {
        "request_id": request_id,
        "label_source": "automated_eval",
        "metric": "faithfulness",
        "metric_version": "ragas-0.4.0",
        "score": score,
        "accepted": score >= 0.5,
        "attribution": "direct",
        "evaluator": "mezvuat-eval/ragas-0.4",
        "eval_run_id": "mlflow:abc123",
        "dataset_version": "eval-2026-11",
        "labelled_at": datetime(2026, 11, 17, tzinfo=UTC),
        "task_type": "evidence_synthesis",
        **overrides,
    }
    return FeedbackItem(**values)


def batch_result(body: dict[str, Any], statuses: list[str] | None = None) -> httpx.Response:
    results = []
    for index, item in enumerate(body["items"]):
        status = statuses[index] if statuses else "stored"
        feedback = None
        if status in ("stored", "replayed"):
            feedback = {
                "id": "fb_" + mmr_id()[4:],
                "request_id": item["request_id"],
                "status": status,
                "training_eligible": item.get("attribution", "direct") == "direct",
                "created_at": "2026-11-17T12:00:00Z",
            }
        results.append({"index": index, "status": status, "feedback": feedback, "error": None})
    response = {"results": results}
    assert_valid("FeedbackBatchResponse", response)
    return httpx.Response(200, json=response)


def test_feedback_batches_are_split_at_500_and_valid(router: respx.Router) -> None:
    sent: list[dict[str, Any]] = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        assert_valid("FeedbackBatchRequest", body)
        assert http_request.headers["authorization"] == "Bearer mm_test_key"
        sent.append(body)
        return batch_result(body)

    router.post(f"{BASE}/v1/feedback/batch").mock(side_effect=handle)
    items = [label(mmr_id()) for _ in range(BATCH_LIMIT + 1)]
    outcome = ops(FakeTime()).post_feedback(items)
    assert [len(b["items"]) for b in sent] == [BATCH_LIMIT, 1]
    assert outcome.statuses["stored"] == BATCH_LIMIT + 1
    assert outcome.ok
    first = sent[0]["items"][0]
    assert first["metric_version"] == "ragas-0.4.0"
    assert first["attribution"] == "direct"
    assert first["eval_run_id"] == "mlflow:abc123"
    assert first["notes"] is None


def test_feedback_replays_are_fine_and_conflicts_are_logged_not_overwritten(
    router: respx.Router, caplog: pytest.LogCaptureFixture
) -> None:
    calls = []

    def handle(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        calls.append(body)
        return batch_result(body, ["replayed", "conflict", "not_found"])

    router.post(f"{BASE}/v1/feedback/batch").mock(side_effect=handle)
    with caplog.at_level(logging.WARNING):
        outcome = ops(FakeTime()).post_feedback([label(mmr_id()), label(mmr_id(), 0.2), label(mmr_id())])
    assert outcome.conflicts == [1]
    assert outcome.statuses == {"replayed": 1, "conflict": 1, "not_found": 1}
    assert not outcome.ok
    assert len(calls) == 1  # a conflict is never retried with a different score
    assert "not overwritten" in caplog.text


def test_feedback_429_waits_and_5xx_retries(router: respx.Router) -> None:
    fake = FakeTime()
    responses = iter([error(429, "rpm", retry_after=3), error(503)])

    def handle(http_request: httpx.Request) -> httpx.Response:
        return next(responses, None) or batch_result(json.loads(http_request.content))

    router.post(f"{BASE}/v1/feedback/batch").mock(side_effect=handle)
    outcome = ops(fake).post_feedback([label(mmr_id())])
    assert outcome.ok
    assert fake.slept[0] == 3.0


def test_feedback_down_raises_unavailable(router: respx.Router) -> None:
    router.post(f"{BASE}/v1/feedback/batch").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(ModelMeshUnavailableError):
        ops(FakeTime()).post_feedback([label(mmr_id())])


def test_feedback_validation_matches_the_spec() -> None:
    with pytest.raises(ValueError, match="mmr_"):
        label("req_123")
    with pytest.raises(ValueError):
        label(mmr_id(), score=1.5)
    with pytest.raises(ValueError):
        label(mmr_id(), metric="Faithfulness")
    with pytest.raises(ValueError):
        label(mmr_id(), notes="the document said ...")
    assert_valid("FeedbackRequest", label(mmr_id()).wire())


def test_user_thumbs_are_workflow_user_signals() -> None:
    rid = mmr_id()
    items = user_thumbs_items(
        [rid, rid, "not-an-mmr-id"], rating="down", labelled_at=datetime(2026, 11, 17, tzinfo=UTC)
    )
    assert len(items) == 1
    wire = items[0].wire()
    assert_valid("FeedbackRequest", wire)
    assert wire["label_source"] == "user_signal"
    assert wire["attribution"] == "workflow"
    assert wire["metric"] == "user_thumbs"
    assert wire["score"] == 0.0 and wire["accepted"] is False


def test_delete_feedback_and_404(router: respx.Router) -> None:
    rid, gone = mmr_id(), mmr_id()
    deleted = {"deleted": 4}
    assert_valid("FeedbackDeleteResponse", deleted)
    router.delete(f"{BASE}/v1/feedback/{rid}").mock(return_value=httpx.Response(200, json=deleted))
    router.delete(f"{BASE}/v1/feedback/{gone}").mock(return_value=error(404, "request_not_found"))
    client = ops(FakeTime())
    assert client.delete_feedback(rid) == 4
    assert client.delete_feedback(gone) == 0
    with pytest.raises(ValueError):
        client.delete_feedback("../../v1/cache")


def test_cache_invalidation_body_is_valid(router: respx.Router) -> None:
    def handle(http_request: httpx.Request) -> httpx.Response:
        body = json.loads(http_request.content)
        assert_valid("CacheInvalidateRequest", body)
        assert body == {"namespace": "ws_0123456789abcdef0123456789abcdef"}
        return httpx.Response(200, json={"deleted_entries": 7})

    router.post(f"{BASE}/v1/cache/invalidate").mock(side_effect=handle)
    client = ops(FakeTime())
    assert client.invalidate_cache(namespace="ws_0123456789abcdef0123456789abcdef") == 7
    with pytest.raises(ValueError):
        client.invalidate_cache()
    with pytest.raises(ValueError):
        client.invalidate_cache(namespace="has spaces")


def test_rejections_are_typed(router: respx.Router) -> None:
    router.post(f"{BASE}/v1/cache/invalidate").mock(return_value=error(401, "invalid_api_key"))
    with pytest.raises(InferenceRejectedError, match="401"):
        ops(FakeTime()).invalidate_cache(namespace="ws_x")
