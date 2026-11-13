"""Direct backend tests (ADR-0011) against mocked HTTP. No request leaves the process: respx blocks unmocked calls."""

from __future__ import annotations

import json
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
    InferenceNotPermittedError,
    InferenceRateLimitedError,
    InferenceRejectedError,
    InferenceUnavailableError,
    LLMRequest,
    OpenAICompatibleBackend,
    StreamDelta,
    StreamDone,
    StreamIncompleteError,
    StructuredOutputInvalidError,
    build_llm_client,
    portability_problems,
    portable_schema,
)
from mezvuat.orchestration.models import Draft, QueryAnalysis, Reformulation, ResearchPlan, Sufficiency

GROQ = "https://api.groq.com/openai/v1/chat/completions"
OLLAMA = "http://127.0.0.1:11434/v1/chat/completions"


class Answer(BaseModel):
    value: str
    score: float | None


def request(**overrides: Any) -> LLMRequest:
    base: dict[str, Any] = {
        "task_type": "query_classification",
        "messages": [ChatMessage("system", "s"), ChatMessage("user", "SECRET DOCUMENT TEXT")],
        "correlation": Correlation("run_1", "understand_query", "understand_query@1"),
        "data_policy": "any",
        "max_tokens": 100,
        "response_schema": Answer,
        "timeout_s": 30.0,
    }
    base.update(overrides)
    return LLMRequest(**base)


class FakeTime:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


@pytest.fixture
def fake_time() -> FakeTime:
    return FakeTime()


def groq(fake_time: FakeTime, http: httpx.Client | None = None) -> OpenAICompatibleBackend:
    return OpenAICompatibleBackend(
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        locality="cloud",
        default_model="small",
        models={"evidence_synthesis": "large"},
        api_key="test-key",
        extra_body={"reasoning_effort": "low"},
        max_tokens_field="max_completion_tokens",
        http=http or httpx.Client(),
        sleep=fake_time.sleep,
        clock=fake_time.clock,
        jitter=lambda: 1.0,
    )


def completion(content: str, *, finish: str = "stop") -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "model": "small",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
        "usage": {"prompt_tokens": 12, "completion_tokens": 5},
    }


@respx.mock
def test_structured_request_shape_and_parsing(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(return_value=httpx.Response(200, json=completion('{"value": "x", "score": null}')))
    result = groq(fake_time).complete(request())
    assert isinstance(result.parsed, Answer) and result.parsed.value == "x"
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "small"
    assert sent["max_completion_tokens"] == 100 and "max_tokens" not in sent
    assert sent["reasoning_effort"] == "low"
    fmt = sent["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"]["score"]["type"] == ["number", "null"]
    assert route.calls[0].request.headers["authorization"] == "Bearer test-key"
    assert result.attempts[0].client_request_id == route.calls[0].request.headers["x-request-id"]
    assert result.usage.prompt_tokens == 12
    assert result.cost_estimated_usd is None and result.cache_status is None


@respx.mock
def test_task_type_selects_the_configured_model(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(return_value=httpx.Response(200, json=completion('{"value": "x", "score": 1}')))
    groq(fake_time).complete(request(task_type="evidence_synthesis"))
    assert json.loads(route.calls[0].request.content)["model"] == "large"


@respx.mock
def test_invalid_output_gets_one_repair_retry_with_the_error(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(
        side_effect=[
            httpx.Response(200, json=completion('{"value": 3}')),
            httpx.Response(200, json=completion('{"value": "fixed", "score": 0.5}')),
        ]
    )
    result = groq(fake_time).complete(request())
    assert isinstance(result.parsed, Answer) and result.parsed.value == "fixed"
    assert [a.status for a in result.attempts] == ["invalid_output", "ok"]
    repair = json.loads(route.calls[1].request.content)["messages"]
    assert repair[-1]["role"] == "user" and "failed validation" in repair[-1]["content"]
    ids = {a.client_request_id for a in result.attempts}
    assert len(ids) == 2, "a fresh X-Request-ID per HTTP attempt"


@respx.mock
def test_repair_exhausted_raises_with_both_attempts(fake_time: FakeTime) -> None:
    respx.post(GROQ).mock(return_value=httpx.Response(200, json=completion("not json")))
    with pytest.raises(StructuredOutputInvalidError) as caught:
        groq(fake_time).complete(request())
    assert len(caught.value.attempts) == 2


@respx.mock
def test_no_repair_when_the_budget_allows_none(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(return_value=httpx.Response(200, json=completion("not json")))
    with pytest.raises(StructuredOutputInvalidError):
        groq(fake_time).complete(request(repair_attempts=0))
    assert route.call_count == 1


@respx.mock
def test_429_waits_for_retry_after_when_it_fits(fake_time: FakeTime) -> None:
    respx.post(GROQ).mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"type": "rate_limit"}}),
            httpx.Response(200, json=completion('{"value": "x", "score": 1}')),
        ]
    )
    result = groq(fake_time).complete(request())
    assert fake_time.slept == [2.0]
    assert [a.http_status for a in result.attempts] == [429, 200]


@respx.mock
def test_429_beyond_the_deadline_fails_without_waiting(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(return_value=httpx.Response(429, headers={"retry-after": "600"}))
    with pytest.raises(InferenceRateLimitedError) as caught:
        groq(fake_time).complete(request(timeout_s=30.0))
    assert route.call_count == 1
    assert fake_time.slept == []
    assert caught.value.code == "inference_rate_limited"


@respx.mock
def test_5xx_retries_twice_then_reports_unavailable(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(return_value=httpx.Response(503, json={"error": {"type": "overloaded"}}))
    with pytest.raises(InferenceUnavailableError) as caught:
        groq(fake_time).complete(request())
    assert route.call_count == 3
    assert fake_time.slept == [0.5, 1.0]
    assert len(caught.value.attempts) == 3


@respx.mock
def test_transport_errors_retry_then_report_unavailable(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(InferenceUnavailableError):
        groq(fake_time).complete(request())
    assert route.call_count == 3


@respx.mock
def test_4xx_is_not_retried_and_does_not_echo_the_body(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(
        return_value=httpx.Response(
            400, json={"error": {"type": "invalid_request_error", "code": "bad", "message": "SECRET DOCUMENT TEXT"}}
        )
    )
    with pytest.raises(InferenceRejectedError) as caught:
        groq(fake_time).complete(request())
    assert route.call_count == 1
    assert "SECRET" not in str(caught.value)
    assert "SECRET" not in (caught.value.attempts[0].error_code or "")


@respx.mock
def test_cloud_backend_never_sends_local_only_data(fake_time: FakeTime) -> None:
    route = respx.post(GROQ).mock(return_value=httpx.Response(200, json=completion("{}")))
    with pytest.raises(InferenceNotPermittedError):
        groq(fake_time).complete(request(data_policy="local_only"))
    with pytest.raises(InferenceNotPermittedError):
        list(groq(fake_time).stream(request(data_policy="local_only", response_schema=None)))
    assert route.call_count == 0


@respx.mock
def test_direct_client_routes_by_data_policy() -> None:
    groq_route = respx.post(GROQ).mock(return_value=httpx.Response(200, json=completion('{"value": "g", "score": 1}')))
    ollama_route = respx.post(OLLAMA).mock(
        return_value=httpx.Response(200, json=completion('{"value": "o", "score": 1}'))
    )
    settings = Settings(groq_api_key="k", ollama_base_url="http://127.0.0.1:11434", direct_public_backend="groq")
    client = build_llm_client(settings)
    public = client.complete(request(data_policy="any"))
    private = client.complete(request(data_policy="local_only"))
    assert (public.provider, private.provider) == ("groq", "ollama")
    assert groq_route.call_count == 1 and ollama_route.call_count == 1
    assert "authorization" not in ollama_route.calls[0].request.headers


@respx.mock
def test_private_workspace_without_a_local_backend_is_refused() -> None:
    route = respx.post(GROQ).mock(return_value=httpx.Response(200, json=completion("{}")))
    client = build_llm_client(Settings(groq_api_key="k", ollama_base_url=None))
    with pytest.raises(InferenceNotPermittedError):
        client.complete(request(data_policy="local_only"))
    assert route.call_count == 0


def test_no_backend_configured_is_a_clear_error() -> None:
    client = DirectClient(public=None, local=None)
    with pytest.raises(InferenceNotPermittedError, match="no inference backend"):
        client.complete(request())


def test_modelmesh_mode_is_not_available_before_m8() -> None:
    with pytest.raises(InferenceRejectedError, match="M8"):
        build_llm_client(Settings(inference_mode="modelmesh"))


def sse(*events: Any, done: bool = True) -> bytes:
    lines = [": keep-alive"]
    lines += [f"data: {json.dumps(e)}" for e in events]
    if done:
        lines.append("data: [DONE]")
    return ("\n\n".join(lines) + "\n\n").encode()


def delta(text: str, finish: str | None = None) -> dict[str, Any]:
    return {
        "id": "c1",
        "model": "large",
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}],
    }


def streamed(fake_time: FakeTime, body: bytes) -> Iterator[Any]:
    respx.post(GROQ).mock(return_value=httpx.Response(200, content=body, headers={"content-type": "text/event-stream"}))
    return groq(fake_time).stream(request(task_type="final_response", response_schema=None))


@respx.mock
def test_stream_yields_deltas_then_a_result_with_usage(fake_time: FakeTime) -> None:
    usage_chunk = {"id": "c1", "choices": [], "usage": {"prompt_tokens": 40, "completion_tokens": 3}}
    events = list(streamed(fake_time, sse(delta("Hello "), delta("world", "stop"), usage_chunk)))
    assert [e.text for e in events if isinstance(e, StreamDelta)] == ["Hello ", "world"]
    done = events[-1]
    assert isinstance(done, StreamDone)
    assert done.result.text == "Hello world" and done.result.finish_reason == "stop"
    assert done.result.usage.prompt_tokens == 40
    sent = json.loads(respx.calls[0].request.content)
    assert sent["stream"] is True and sent["stream_options"] == {"include_usage": True}


@respx.mock
def test_stream_without_done_is_incomplete(fake_time: FakeTime) -> None:
    with pytest.raises(StreamIncompleteError) as caught:
        list(streamed(fake_time, sse(delta("partial"), done=False)))
    assert caught.value.attempts[-1].error_code == "stream_incomplete"


@respx.mock
def test_stream_error_event_is_incomplete(fake_time: FakeTime) -> None:
    with pytest.raises(StreamIncompleteError):
        list(streamed(fake_time, sse(delta("partial"), {"error": {"code": "stream_interrupted"}}, done=False)))


@respx.mock
def test_stream_pre_commit_http_error_is_handled_like_any_error(fake_time: FakeTime) -> None:
    respx.post(GROQ).mock(return_value=httpx.Response(400, json={"error": {"type": "invalid_request_error"}}))
    with pytest.raises(InferenceRejectedError):
        list(groq(fake_time).stream(request(task_type="final_response", response_schema=None)))


@pytest.mark.parametrize("model", [QueryAnalysis, ResearchPlan, Sufficiency, Reformulation, Draft])
def test_every_node_schema_is_in_the_portable_subset(model: type[BaseModel]) -> None:
    schema = portable_schema(model)
    assert portability_problems(schema) == []
    text = json.dumps(schema)
    for keyword in ("$ref", "anyOf", "maxLength", "minLength", "pattern", "oneOf"):
        assert keyword not in text


def test_non_portable_schemas_are_reported() -> None:
    problems = portability_problems({"type": "object", "properties": {"a": {"type": "string", "maxLength": 3}}})
    assert any("maxLength" in p for p in problems)
    assert any("additionalProperties" in p for p in problems)
    assert any("required" in p for p in problems)
