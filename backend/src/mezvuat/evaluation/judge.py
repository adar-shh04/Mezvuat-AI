"""The evaluation judge (ADR-0012, contract §2.2). Pluggable: anything with `model` and `structured()` is a judge.

Production judges go through Mezvuat's own `LLMClient`, so ModelMesh extensions are sent by the same code as every
other call (instead of an httpx hook on RAGAS's OpenAI client, as ADR-0012 first planned):
- `task_type: evaluation_judge`, a **pinned** model, `routing.fallback: "none"` (pinned requests never fall back),
  temperature 0 and a fixed seed, so `cache.mode` is `exact` and re-scoring is reproducible;
- if the pinned judge is unavailable, the call raises and the evaluation run fails. It never switches judges;
- decision D1: judge calls carry evidence, so they run only on public-corpus items (`data_policy: "any"`); the
  evaluation runner refuses private workspaces before any judge call.
Tests use fakes; nothing here has been run against a real judge yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

from mezvuat.core.config import Settings
from mezvuat.inference import ChatMessage, Correlation, InferenceRejectedError, LLMClient, LLMRequest
from mezvuat.inference.direct import DirectClient, OpenAICompatibleBackend

T = TypeVar("T", bound=BaseModel)
JUDGE_PROMPT_VERSION = "ragas-0.4"


class Judge(Protocol):
    model: str

    def structured(self, prompt: str, response_model: type[T]) -> T: ...


@dataclass
class LLMClientJudge:
    llm: LLMClient
    model: str
    eval_run_id: str
    max_tokens: int = 1024
    timeout_s: float = 60.0

    def structured(self, prompt: str, response_model: type[T]) -> T:
        request = LLMRequest(
            task_type="evaluation_judge",
            messages=[ChatMessage("user", prompt)],
            correlation=Correlation(f"eval_{self.eval_run_id}", "judge", JUDGE_PROMPT_VERSION),
            data_policy="any",
            max_tokens=self.max_tokens,
            response_schema=response_model,
            temperature=0.0,
            seed=0,
            timeout_s=self.timeout_s,
            model=self.model,
        )
        result = self.llm.complete(request)
        if not isinstance(result.parsed, response_model):
            raise InferenceRejectedError("the judge returned no parsed output", code="judge_output_missing")
        return result.parsed


def judge_from_settings(settings: Settings, model: str, eval_run_id: str, backend: str = "direct") -> LLMClientJudge:
    """`direct`: Groq with the model pinned (free tier, D4). `modelmesh`: ModelMesh with an explicit model id
    (`groq/<model>` style), `routing.fallback: "none"`. The judge's family must differ from the generator's."""
    if backend == "modelmesh":
        from mezvuat.inference.factory import modelmesh_client

        client = modelmesh_client(settings.model_copy(update={"inference_mode": "modelmesh"}))
        return LLMClientJudge(client, model, eval_run_id)
    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
        raise InferenceRejectedError("the direct judge needs GROQ_API_KEY", code="judge_not_configured")
    groq = OpenAICompatibleBackend(
        provider="groq",
        base_url=settings.groq_base_url,
        locality="cloud",
        default_model=model.removeprefix("groq/"),
        api_key=settings.groq_api_key.get_secret_value(),
        extra_body={"reasoning_effort": "low"},
        max_tokens_field="max_completion_tokens",
        http=httpx.Client(timeout=90.0),
    )
    return LLMClientJudge(DirectClient(public=groq, local=None), model, eval_run_id)
