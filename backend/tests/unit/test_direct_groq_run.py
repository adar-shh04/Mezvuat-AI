"""D11 (direct half): a full research run with only GROQ_API_KEY configured.

The Groq endpoint is mocked with canned responses in Groq's OpenAI-compatible wire format (respx blocks every
other request), so this proves the wiring: settings -> DirectClient -> Groq backend -> graph -> cited answer.
It is not a live call; the live smoke test is `scripts/smoke_run.py` and is run by hand.
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import respx

from mezvuat.core.config import Settings
from mezvuat.inference import build_llm_client
from mezvuat.orchestration import Budget, ResearchState, WorkflowDeps, build_graph, run_to_end
from tests.fakes import KeywordSearch
from tests.unit.test_research_graph import CHUNKS, PAGES, analysis, coverage

GROQ = "https://api.groq.com/openai/v1/chat/completions"


def _content(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.content)


def _evidence_id(prompt: str, phrase: str) -> str:
    for match in re.finditer(r'<evidence id="(E\d+)"[^>]*>(.*?)</evidence', prompt, re.DOTALL):
        if phrase in match.group(2):
            return match.group(1)
    raise AssertionError(phrase)


def groq_reply(request: httpx.Request) -> httpx.Response:
    body = _content(request)
    prompt = body["messages"][-1]["content"]
    if body.get("stream"):
        marker = re.search(r"\[(E\d+)\]", prompt.split("Verified claims:", 1)[1])
        assert marker is not None
        text = f"BM25 weighs term frequency against document frequency [{marker.group(1)}]."
        chunks = [
            {"id": "x", "model": body["model"], "choices": [{"index": 0, "delta": {"content": text}}]},
            {"id": "x", "model": body["model"], "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"id": "x", "choices": [], "usage": {"prompt_tokens": 300, "completion_tokens": 20}},
        ]
        stream = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(200, content=stream.encode(), headers={"content-type": "text/event-stream"})
    name = body["response_format"]["json_schema"]["name"]
    if name == "QueryAnalysis":
        payload: dict[str, Any] = analysis(single_hop=True)
    elif name == "Sufficiency":
        payload = coverage(True)
    elif name == "Draft":
        evidence_id = _evidence_id(prompt, "BM25 is a bag-of-words")
        payload = {
            "claims": [
                {
                    "id": "C1",
                    "text": "BM25 scores documents by term frequency and inverse document frequency.",
                    "evidence_ids": [evidence_id],
                    "quotes": [
                        {
                            "evidence_id": evidence_id,
                            "text": "It scores documents by term frequency and inverse document frequency",
                        }
                    ],
                    "kind": "fact",
                }
            ],
            "conflicts": [],
            "gaps": [],
        }
    else:
        raise AssertionError(f"unexpected schema {name}")
    return httpx.Response(
        200,
        headers={"x-request-id": "req_groq_1"},
        json={
            "id": "chatcmpl-1",
            "model": body["model"],
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": json.dumps(payload)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 200, "completion_tokens": 40},
        },
    )


@respx.mock
def test_full_run_with_only_a_groq_key() -> None:
    route = respx.post(GROQ).mock(side_effect=groq_reply)
    settings = Settings(groq_api_key="gsk-test-placeholder", ollama_base_url=None, inference_mode="direct")
    llm = build_llm_client(settings)
    graph = build_graph(WorkflowDeps(llm=llm, search=KeywordSearch(CHUNKS), pages=PAGES))
    initial = ResearchState(
        run_id="run_01D11",
        workspace_id="ws_1",
        question="How does BM25 rank documents?",
        data_policy="any",
        budget=Budget(llm_calls_left=12, tokens_left=100_000, deadline=1e12),
    )
    final = run_to_end(graph, initial)
    assert final.status == "completed", final.error
    answer = final.answer
    assert answer is not None and answer.rendering == "model"
    assert answer.citations and answer.citations[0].quote.startswith("It scores documents")
    assert route.call_count == 4
    assert {c.provider for c in final.llm_calls} == {"groq"}
    sent = [_content(call.request) for call in route.calls]
    assert [s["model"] for s in sent] == [
        settings.groq_model_small,
        settings.groq_model_small,
        settings.groq_model_large,
        settings.groq_model_large,
    ]
    assert all(call.request.headers["authorization"] == "Bearer gsk-test-placeholder" for call in route.calls)
    assert final.llm_calls[0].provider_request_id == "req_groq_1"
