"""ModelMesh client behaviour (M8, ADR-0011, contract 1.0) against a mocked ModelMesh built from openapi-v1.yaml.

respx blocks every unmocked request, so nothing here can reach a real ModelMesh, Groq or Ollama.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from typing import Any

import httpx
import pytest
import respx
from pydantic import BaseModel

from mezvuat.core.config import Settings
from mezvuat.inference import (
    ChatMessage,
    Correlation,
    DirectClient,
    InferenceContractMismatchError,
    InferenceNotPermittedError,
    InferenceRateLimitedError,
    InferenceRejectedError,
    LLMRequest,
    ModelMeshRejectedError,
    ModelMeshUnavailableError,
    OpenAICompatibleBackend,
    StreamDelta,
    StreamDone,
    StreamIncompleteError,
    build_llm_client,
)
from mezvuat.inference.modelmesh import (
    CircuitBreaker,
    ContractNegotiator,
    ModelMeshBackend,
    ModelMeshClient,
    cache_namespace,
    context_fingerprint,
    evidence_key,
    run_trace_id,
)
from tests.modelmesh_stub import (
    BASE,
    TRACE_ID,
    MeshStub,
    chat_error_statuses,
    chunk,
    completion,
    error,
    interrupted_event,
    mmr_id,
    sse,
    text_stream,
)

GROQ = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"
SECRET = b"test-namespace-secret"
TRACEPARENT = re.compile(r"^00-([0-9a-f]{32})-([0-9a-f]{16})-01$")


class Answer(BaseModel):
    value: str


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def request(**overrides: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "task_type": "query_classification",
        "messages": [ChatMessage("system", "s"), ChatMessage("user", "q")],
        "correlation": Correlation("run_01JTESTRUN", "understand_query", "understand_query@1"),
        "data_policy": "any",
        "max_tokens": 100,
        "response_schema": Answer,
        "timeout_s": 30.0,
        "seed": 0,
        "workspace_id": "ws_public",
    }
    base.update(overrides)
    return LLMRequest(**base)


def openai_completion(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "model": "openai/gpt-oss-20b",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        },
    )


def direct(fake: FakeTime, *, ollama: bool = True) -> DirectClient:
    http = httpx.Client()
    groq = OpenAICompatibleBackend(
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        locality="cloud",
        default_model="openai/gpt-oss-20b",
        api_key="test-key",
        http=http,
        sleep=fake.sleep,
        clock=fake.clock,
        jitter=lambda: 0.5,
    )
    local = (
        OpenAICompatibleBackend(
            provider="ollama",
            base_url="http://127.0.0.1:11434/v1",
            locality="local",
            default_model="llama3.2:3b",
            http=http,
            sleep=fake.sleep,
            clock=fake.clock,
            jitter=lambda: 0.5,
        )
        if ollama
        else None
    )
    return DirectClient(public=groq, local=local)


def mesh(
    fake: FakeTime,
    *,
    fallback: DirectClient | None = None,
    min_quality: dict[str, float] | None = None,
    threshold: int = 5,
) -> ModelMeshClient:
    http = httpx.Client()
    negotiator = ContractNegotiator(base_url=BASE, http=http, api_key="mm_test_key", clock=fake.clock)
    backend = ModelMeshBackend(
        base_url=BASE,
        negotiator=negotiator,
        namespace_secret=SECRET,
        api_key="mm_test_key",
        min_quality=min_quality or {},
        http=http,
        sleep=fake.sleep,
        clock=fake.clock,
        jitter=lambda: 0.5,
    )

    def probe() -> bool:
        return http.get(f"{BASE}/v1/health").status_code == 200

    breaker = CircuitBreaker(probe=probe, threshold=threshold, open_s=30.0, clock=fake.clock)
    return ModelMeshClient(backend=backend, breaker=breaker, fallback=fallback)


@pytest.fixture
def fake() -> FakeTime:
    return FakeTime()


@pytest.fixture
def router() -> Iterator[respx.Router]:
    with respx.mock(assert_all_called=False) as mocked:
        yield mocked


@pytest.fixture
def stub(router: respx.Router) -> MeshStub:
    return MeshStub(router)


ANSWER = json.dumps({"value": "ok"})


# ----- negotiation (contract §0, §2.4) ------------------------------------------------------------------------------


def test_contract_is_read_once_and_cached(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [completion(ANSWER), completion(ANSWER)]
    client = mesh(fake)
    client.complete(request())
    client.complete(request())
    assert stub.contract_calls == 1
    assert stub.calibration_calls == 1
    fake.now += 601  # the contract is refreshed after ten minutes
    stub.chat.append(completion(ANSWER))
    client.complete(request())
    assert stub.contract_calls == 2


def test_only_extensions_the_server_lists_are_sent(stub: MeshStub, fake: FakeTime) -> None:
    stub.without_extension("routing.timeout_ms")
    stub.without_extension("cache.context_fingerprint")
    stub.without_extension("routing.policy")
    stub.chat = [completion(ANSWER)]
    mesh(fake).complete(request())
    body = stub.bodies[0]
    assert "timeout_ms" not in body["routing"]
    assert "policy" not in body["routing"]
    assert "context_fingerprint" not in body["cache"]
    assert body["routing"]["data_policy"] == "any"
    assert body["cache"]["namespace"].startswith("ws_")


@pytest.mark.parametrize("missing", ["routing.data_policy", "cache.namespace"])
def test_missing_hard_extension_fails_without_sending(stub: MeshStub, fake: FakeTime, missing: str) -> None:
    stub.without_extension(missing)
    client = mesh(fake, fallback=direct(fake))
    with pytest.raises(InferenceContractMismatchError, match=missing):
        client.complete(request())
    with pytest.raises(InferenceContractMismatchError):
        client.negotiate()  # the start-up check is a hard failure too
    assert stub.bodies == []


def test_other_major_contract_version_is_a_mismatch(stub: MeshStub, fake: FakeTime) -> None:
    stub.contract["contract_version"] = "2.0"
    with pytest.raises(InferenceContractMismatchError, match=r"2\.0"):
        mesh(fake).complete(request())
    assert stub.bodies == []


def test_unreachable_at_start_up_is_logged_not_raised(
    router: respx.Router, fake: FakeTime, caplog: pytest.LogCaptureFixture
) -> None:
    router.get(f"{BASE}/v1/contract").mock(side_effect=httpx.ConnectError("refused"))
    with caplog.at_level(logging.WARNING):
        mesh(fake).negotiate()
    assert "unreachable at start-up" in caplog.text


def test_min_quality_only_for_calibrated_task_types(stub: MeshStub, fake: FakeTime) -> None:
    quality = {"query_classification": 0.7, "evidence_assessment": 0.8}
    # The spec's example marks everything uncalibrated: nothing is sent.
    stub.chat = [completion(ANSWER)]
    mesh(fake, min_quality=quality).complete(request())
    assert "min_quality" not in stub.bodies[-1]["routing"]
    # Calibrated, with a configured value: sent.
    stub.calibrate("query_classification", calibrated=True)
    stub.chat = [completion(ANSWER), completion(ANSWER)]
    client = mesh(fake, min_quality=quality)
    client.complete(request())
    assert stub.bodies[-1]["routing"]["min_quality"] == 0.7
    # Calibrated for another task type only: not sent for this one.
    client.complete(request(task_type="evidence_assessment"))
    assert "min_quality" not in stub.bodies[-1]["routing"]


def test_calibration_failure_means_no_min_quality(stub: MeshStub, fake: FakeTime) -> None:
    stub.calibrate("query_classification", calibrated=True)
    stub.calibration_status = 500
    stub.chat = [completion(ANSWER)]
    mesh(fake, min_quality={"query_classification": 0.7}).complete(request())
    assert "min_quality" not in stub.bodies[0]["routing"]


def test_unknown_task_type_is_sent_as_general(stub: MeshStub, fake: FakeTime) -> None:
    stub.contract["task_types"] = [t for t in stub.contract["task_types"] if t != "evidence_assessment"]
    stub.chat = [completion(ANSWER)]
    mesh(fake).complete(request(task_type="evidence_assessment"))
    metadata = stub.bodies[0]["metadata"]
    assert metadata["task_type"] == "general"
    assert metadata["task_type_intended"] == "evidence_assessment"


def test_unknown_parameter_refreshes_the_contract_and_resends_once(stub: MeshStub, fake: FakeTime) -> None:
    def drop_ttl_then_reject(body: dict[str, Any]) -> httpx.Response:
        stub.without_extension("routing.timeout_ms")
        return error(400, "unknown_parameter")

    stub.chat = [drop_ttl_then_reject, completion(ANSWER)]
    result = mesh(fake).complete(request())
    assert result.backend == "modelmesh"
    assert stub.contract_calls == 2
    assert "timeout_ms" in stub.bodies[0]["routing"]
    assert "timeout_ms" not in stub.bodies[1]["routing"]


# ----- request format (contract §1, §2, §2.3) -----------------------------------------------------------------------


def test_metadata_is_flat_strings_with_a_valid_correlation_id(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [completion(ANSWER)]
    mesh(fake).complete(request())
    metadata = stub.bodies[0]["metadata"]
    assert metadata == {
        "application": "mezvuat",
        "task_type": "query_classification",
        "correlation_id": "run_01JTESTRUN",
        "workflow_node": "understand_query",
        "prompt_version": "understand_query@1",
        "iteration": "0",
    }
    assert all(isinstance(v, str) for v in metadata.values())
    assert re.fullmatch(r"[A-Za-z0-9_:.-]{1,128}", metadata["correlation_id"])


@pytest.mark.parametrize("run_id", ["run with spaces", "x" * 129, "run/1", ""])
def test_invalid_correlation_id_is_rejected_before_sending(stub: MeshStub, fake: FakeTime, run_id: str) -> None:
    with pytest.raises(InferenceRejectedError) as raised:
        mesh(fake, fallback=direct(fake)).complete(
            request(correlation=Correlation(run_id, "understand_query", "understand_query@1"))
        )
    assert raised.value.code == "invalid_correlation_id"
    assert stub.bodies == []


def test_namespace_and_fingerprint_formats(stub: MeshStub, fake: FakeTime) -> None:
    keys = (evidence_key("c1", "v1", "alpha"), evidence_key("c2", "v1", "beta"))
    stub.chat = [completion(ANSWER)]
    mesh(fake).complete(request(evidence_keys=keys))
    cache = stub.bodies[0]["cache"]
    assert re.fullmatch(r"ws_[0-9a-f]{32}", cache["namespace"])
    assert re.fullmatch(r"sha256:[0-9a-f]{64}", cache["context_fingerprint"])
    assert cache["namespace"] == cache_namespace(SECRET, "ws_public")
    assert "ws_public" not in json.dumps(stub.bodies[0])  # the raw workspace id never leaves
    assert cache["context_fingerprint"] == context_fingerprint("ws_public", "understand_query@1", keys)


def test_fingerprint_ignores_order_and_tracks_content() -> None:
    a, b = evidence_key("c1", "v1", "alpha"), evidence_key("c2", "v1", "beta")
    base = context_fingerprint("ws_1", "p@1", [a, b])
    assert base == context_fingerprint("ws_1", "p@1", [b, a])
    assert base != context_fingerprint("ws_1", "p@1", [a, evidence_key("c2", "v1", "beta, edited")])
    assert base != context_fingerprint("ws_1", "p@1", [a, evidence_key("c2", "v2", "beta")])
    assert base != context_fingerprint("ws_2", "p@1", [a, b])
    assert base != context_fingerprint("ws_1", "p@2", [a, b])
    assert cache_namespace(SECRET, "ws_1") != cache_namespace(SECRET, "ws_2")
    assert cache_namespace(SECRET, "ws_1") != cache_namespace(b"other secret", "ws_1")


def test_cache_mode_per_call_kind(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [completion(ANSWER), completion("free text", structured=False)]
    client = mesh(fake)
    client.complete(request())
    client.complete(request(response_schema=None, temperature=0.7, seed=None, task_type="general"))
    assert stub.bodies[0]["cache"]["mode"] == "exact"
    assert stub.bodies[1]["cache"]["mode"] == "off"


def test_data_policy_follows_the_workspace(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [completion(ANSWER), completion(ANSWER)]
    client = mesh(fake)
    client.complete(request(data_policy="any"))
    client.complete(request(data_policy="local_only", workspace_id="ws_private"))
    assert [b["routing"]["data_policy"] for b in stub.bodies] == ["any", "local_only"]


def test_traceparent_is_sent_and_the_trace_id_is_stored(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [completion(ANSWER), completion(ANSWER)]
    client = mesh(fake)
    first = client.complete(request())
    client.complete(request())
    parents = [TRACEPARENT.match(h["traceparent"]) for h in stub.headers]
    assert all(parents)
    # Every call of one run shares one trace id; each call is its own span.
    assert {p.group(1) for p in parents if p} == {run_trace_id("run_01JTESTRUN")}
    assert len({p.group(2) for p in parents if p}) == 2
    attempt = first.attempts[-1]
    assert attempt.modelmesh_trace_id == TRACE_ID
    assert attempt.modelmesh_request_id is not None and attempt.modelmesh_request_id.startswith("mmr_")
    assert attempt.cost_estimated_usd == 0.00059
    assert attempt.cache_status == "miss"
    assert attempt.backend == "modelmesh"
    assert stub.headers[0]["authorization"] == "Bearer mm_test_key"


def test_each_attempt_gets_a_fresh_request_id(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [error(503), completion(ANSWER)]
    result = mesh(fake).complete(request())
    ids = [h["x-request-id"] for h in stub.headers]
    assert len(set(ids)) == 2
    assert [a.client_request_id for a in result.attempts] == ids


def test_structured_capabilities_and_the_strict_downgrade(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [error(422, "constraints_unsatisfiable"), completion(ANSWER), completion(ANSWER)]
    client = mesh(fake)
    result = client.complete(request(task_type="evidence_synthesis"))
    assert [b["routing"]["required_capabilities"] for b in stub.bodies] == [["json_schema_strict"], ["json_schema"]]
    assert result.structured_mode == "json_schema"
    client.complete(request(task_type="query_classification"))
    assert stub.bodies[-1]["routing"]["required_capabilities"] == ["json_schema"]
    assert stub.bodies[-1]["response_format"]["type"] == "json_schema"


def test_pinned_judge_model_sends_fallback_none_and_never_falls_back(
    stub: MeshStub, router: respx.Router, fake: FakeTime
) -> None:
    groq = router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    stub.chat = [error(503)] * 3
    judge = request(task_type="evaluation_judge", model="groq/openai/gpt-oss-120b", workspace_id=None)
    with pytest.raises(ModelMeshUnavailableError):
        mesh(fake, fallback=direct(fake)).complete(judge)
    body = stub.bodies[0]
    assert body["model"] == "groq/openai/gpt-oss-120b"
    assert body["routing"]["fallback"] == "none"
    assert "policy" not in body["routing"]
    assert "min_quality" not in body["routing"]
    assert not groq.called


def test_policy_defaults_and_override(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [completion(ANSWER), completion(ANSWER), completion(ANSWER)]
    client = mesh(fake)
    client.complete(request(task_type="query_classification"))
    client.complete(request(task_type="evidence_assessment"))
    client.complete(request(task_type="evidence_assessment", routing_policy="latency_optimized"))
    assert [b["routing"]["policy"] for b in stub.bodies] == ["cost_optimized", "quality_balanced", "latency_optimized"]


# ----- fallback table (contract §3.1), one case per status the spec lists ------------------------------------------


FALLBACK = {500, 502, 503, 504}


@pytest.mark.parametrize("status", chat_error_statuses())
def test_fallback_rule_for_every_error_status_in_the_spec(
    stub: MeshStub, router: respx.Router, fake: FakeTime, status: int
) -> None:
    groq = router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    if status == 429:
        stub.chat = [error(429, "rpm", retry_after=2), completion(ANSWER)]
        result = mesh(fake, fallback=direct(fake)).complete(request())
        assert result.backend == "modelmesh"
        assert fake.slept == [2.0]
    elif status in FALLBACK:
        stub.chat = [error(status)] * 3
        result = mesh(fake, fallback=direct(fake)).complete(request())
        assert result.backend == "direct"
        assert result.fallback_reason == f"modelmesh_{status}"
        assert len(stub.bodies) == 3  # two retries before falling back
        assert [a.backend for a in result.attempts] == ["modelmesh"] * 3 + ["direct"]
        assert result.attempts[-1].fallback_reason == f"modelmesh_{status}"
    else:
        stub.chat = [error(status)]
        with pytest.raises(ModelMeshRejectedError) as raised:
            mesh(fake, fallback=direct(fake)).complete(request())
        assert raised.value.status == status
        assert len(stub.bodies) == 1  # never retried
    assert groq.called is (status in FALLBACK)


def test_429_longer_than_the_deadline_fails_without_fallback(
    stub: MeshStub, router: respx.Router, fake: FakeTime
) -> None:
    groq = router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    stub.chat = [error(429, "tpm", retry_after=120)]
    with pytest.raises(InferenceRateLimitedError) as raised:
        mesh(fake, fallback=direct(fake)).complete(request(timeout_s=30.0))
    assert raised.value.attempts[0].http_status == 429
    assert fake.slept == []
    assert not groq.called


def test_repeated_429_fails_after_two_waits(stub: MeshStub, router: respx.Router, fake: FakeTime) -> None:
    groq = router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    stub.chat = [error(429, "rpm", retry_after=1)] * 3
    with pytest.raises(InferenceRateLimitedError):
        mesh(fake, fallback=direct(fake)).complete(request())
    assert fake.slept == [1.0, 1.0]
    assert not groq.called


def test_network_error_retries_once_then_falls_back(stub: MeshStub, router: respx.Router, fake: FakeTime) -> None:
    router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    stub.chat = [httpx.ConnectError("refused"), httpx.ConnectError("refused")]
    result = mesh(fake, fallback=direct(fake)).complete(request())
    assert result.backend == "direct"
    assert result.fallback_reason == "modelmesh_unreachable"


def test_contract_unreachable_falls_back(router: respx.Router, fake: FakeTime) -> None:
    router.get(f"{BASE}/v1/contract").mock(side_effect=httpx.ConnectError("refused"))
    router.get(f"{BASE}/v1/routing/calibration").mock(side_effect=httpx.ConnectError("refused"))
    chat = router.post(f"{BASE}/v1/chat/completions")
    router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    result = mesh(fake, fallback=direct(fake)).complete(request())
    assert result.fallback_reason == "modelmesh_unreachable"
    assert not chat.called


def test_without_fallback_mode_unavailability_surfaces(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [error(503)] * 3
    with pytest.raises(ModelMeshUnavailableError) as raised:
        mesh(fake, fallback=None).complete(request())
    assert raised.value.code == "inference_unavailable"


# ----- decision D1 on fallback ----------------------------------------------------------------------------------------


def test_private_workspace_falls_back_to_local_only(stub: MeshStub, router: respx.Router, fake: FakeTime) -> None:
    groq = router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    ollama = router.post(OLLAMA).mock(return_value=openai_completion(ANSWER))
    stub.chat = [error(503)] * 3
    result = mesh(fake, fallback=direct(fake)).complete(request(data_policy="local_only", workspace_id="ws_private"))
    assert stub.bodies[0]["routing"]["data_policy"] == "local_only"
    assert result.provider == "ollama"
    assert ollama.called
    assert not groq.called


def test_private_workspace_without_a_local_backend_is_not_permitted(
    stub: MeshStub, router: respx.Router, fake: FakeTime
) -> None:
    groq = router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    stub.chat = [error(503)] * 3
    with pytest.raises(InferenceNotPermittedError) as raised:
        mesh(fake, fallback=direct(fake, ollama=False)).complete(
            request(data_policy="local_only", workspace_id="ws_private")
        )
    assert "modelmesh_503" in raised.value.message
    # The ModelMesh attempts are still accounted for.
    assert [a.http_status for a in raised.value.attempts] == [503, 503, 503]
    assert not groq.called


# ----- streaming ------------------------------------------------------------------------------------------------------


def stream_request(**overrides: Any) -> LLMRequest:
    values: dict[str, Any] = {
        "task_type": "final_response",
        "response_schema": None,
        "temperature": 0.2,
        "correlation": Correlation("run_01JTESTRUN", "respond", "respond@1"),
        **overrides,
    }
    return request(**values)


def drain(events: Iterator[Any]) -> tuple[str, Any]:
    text, done = "", None
    for event in events:
        if isinstance(event, StreamDelta):
            text += event.text
        elif isinstance(event, StreamDone):
            done = event.result
    return text, done


def test_stream_sends_include_usage_and_reads_the_usage_chunk(stub: MeshStub, fake: FakeTime) -> None:
    rid = mmr_id()
    stub.chat = [text_stream("BM25 ranks documents.", request_id=rid)]
    text, done = drain(mesh(fake).stream(stream_request()))
    body = stub.bodies[0]
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["cache"]["mode"] == "off"
    assert text == "BM25 ranks documents."
    attempt = done.attempts[-1]
    assert attempt.modelmesh_request_id == rid
    assert attempt.modelmesh_trace_id == TRACE_ID
    assert attempt.usage.prompt_tokens == 120
    assert done.finish_reason == "stop"


def test_stream_failure_before_the_commit_point_falls_back(
    stub: MeshStub, router: respx.Router, fake: FakeTime
) -> None:
    groq_stream = (
        b'data: {"choices":[{"delta":{"content":"direct answer"},"finish_reason":"stop"}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\ndata: [DONE]\n\n'
    )
    groq = router.post(GROQ).mock(
        return_value=httpx.Response(200, content=groq_stream, headers={"content-type": "text/event-stream"})
    )
    stub.chat = [error(503)] * 3
    text, done = drain(mesh(fake, fallback=direct(fake)).stream(stream_request()))
    assert groq.called
    assert text == "direct answer"
    assert done.fallback_reason == "modelmesh_503"
    assert [a.backend for a in done.attempts] == ["modelmesh"] * 3 + ["direct"]


def test_stream_interrupted_raises_incomplete_without_fallback(
    stub: MeshStub, router: respx.Router, fake: FakeTime
) -> None:
    groq = router.post(GROQ).mock(return_value=openai_completion("x"))
    rid = mmr_id()
    stub.chat = [sse([chunk("partial "), interrupted_event(rid)], done=False, request_id=rid)]
    seen: list[str] = []
    with pytest.raises(StreamIncompleteError) as raised:
        for event in mesh(fake, fallback=direct(fake)).stream(stream_request()):
            if isinstance(event, StreamDelta):
                seen.append(event.text)
    assert seen == ["partial "]
    attempt = raised.value.attempts[-1]
    assert attempt.error_code == "stream_interrupted"
    assert attempt.modelmesh_request_id == rid
    assert not groq.called


def test_stream_without_done_is_incomplete(stub: MeshStub, fake: FakeTime) -> None:
    stub.chat = [sse([chunk("all "), chunk("text", finish="stop")], done=False)]
    with pytest.raises(StreamIncompleteError) as raised:
        drain(mesh(fake).stream(stream_request()))
    assert raised.value.attempts[-1].error_code == "stream_incomplete"


# ----- circuit breaker ------------------------------------------------------------------------------------------------


def test_breaker_opens_after_five_failed_calls_and_probes_health(
    stub: MeshStub, router: respx.Router, fake: FakeTime
) -> None:
    groq = router.post(GROQ).mock(return_value=openai_completion(ANSWER))
    client = mesh(fake, fallback=direct(fake))
    stub.chat = [error(503)] * 15
    for _ in range(5):
        assert client.complete(request()).backend == "direct"
    sent = len(stub.bodies)
    assert client.breaker.state == "open"
    # Open: straight to the fallback, ModelMesh is not asked.
    result = client.complete(request())
    assert result.fallback_reason == "circuit_open"
    assert len(stub.bodies) == sent
    # After 30 s a health probe decides. Down: stays open.
    fake.now += 31
    stub.health_status = 503
    assert client.complete(request()).fallback_reason == "circuit_open"
    assert stub.health_calls == 1
    # Up again: half-open, the call goes to ModelMesh, and a success closes the breaker.
    fake.now += 31
    stub.health_status = 200
    stub.chat = [completion(ANSWER)]
    assert client.complete(request()).backend == "modelmesh"
    assert client.breaker.state == "closed"
    assert groq.call_count == 7


def test_4xx_and_429_count_as_healthy(stub: MeshStub, fake: FakeTime) -> None:
    client = mesh(fake, fallback=None, threshold=2)
    stub.chat = [error(503)] * 3 + [error(400, "invalid_value")] + [error(503)] * 3
    with pytest.raises(ModelMeshUnavailableError):
        client.complete(request())
    with pytest.raises(ModelMeshRejectedError):
        client.complete(request())
    with pytest.raises(ModelMeshUnavailableError):
        client.complete(request())
    assert client.breaker.state == "closed"  # the 400 reset the count


# ----- factory --------------------------------------------------------------------------------------------------------


def test_factory_builds_modelmesh_with_fallback(router: respx.Router) -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        inference_mode="modelmesh_with_fallback",
        modelmesh_base_url=f"{BASE}/v1",
        mezvuat_secret_key="s",
        groq_api_key="g",
    )
    client = build_llm_client(settings)
    assert isinstance(client, ModelMeshClient)
    assert client.backend.base_url == BASE
    assert client.fallback is not None and client.fallback.public is not None
    assert not router.calls  # building the client sends nothing


def test_factory_requires_the_namespace_secret() -> None:
    settings = Settings(_env_file=None, inference_mode="modelmesh", modelmesh_base_url=BASE)  # type: ignore[call-arg]
    with pytest.raises(InferenceRejectedError, match="MEZVUAT_SECRET_KEY"):
        build_llm_client(settings)
