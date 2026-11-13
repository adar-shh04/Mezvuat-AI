"""Builds the configured `LLMClient`. Only this module turns settings into provider endpoints."""

from __future__ import annotations

import httpx

from mezvuat.core.config import Settings
from mezvuat.inference.direct import DirectClient, OpenAICompatibleBackend
from mezvuat.inference.errors import InferenceRejectedError

# Larger models for the calls whose quality matters most; the rest use the small, fast model.
_LARGE_TASKS = ("evidence_synthesis", "final_response", "citation_verification")


def groq_backend(settings: Settings, http: httpx.Client | None = None) -> OpenAICompatibleBackend | None:
    if settings.groq_api_key is None or not settings.groq_api_key.get_secret_value():
        return None
    return OpenAICompatibleBackend(
        provider="groq",
        base_url=settings.groq_base_url,
        locality="cloud",
        default_model=settings.groq_model_small,
        models=dict.fromkeys(_LARGE_TASKS, settings.groq_model_large),
        api_key=settings.groq_api_key.get_secret_value(),
        # gpt-oss models reason before answering; low effort keeps latency and tokens within the free tier.
        extra_body={"reasoning_effort": "low"},
        max_tokens_field="max_completion_tokens",
        http=http or httpx.Client(timeout=60.0),
    )


def ollama_backend(settings: Settings, http: httpx.Client | None = None) -> OpenAICompatibleBackend | None:
    if not settings.ollama_base_url:
        return None
    large = settings.ollama_model_large or settings.ollama_model
    return OpenAICompatibleBackend(
        provider="ollama",
        base_url=settings.ollama_base_url.rstrip("/") + "/v1",
        locality="local",
        default_model=settings.ollama_model,
        models=dict.fromkeys(_LARGE_TASKS, large),
        http=http or httpx.Client(timeout=120.0),
    )


def build_llm_client(settings: Settings, http: httpx.Client | None = None) -> DirectClient:
    if settings.inference_mode != "direct":
        raise InferenceRejectedError(
            f"INFERENCE_MODE={settings.inference_mode} needs the ModelMesh backend, which arrives in M8",
            code="inference_mode_unsupported",
        )
    local = ollama_backend(settings, http)
    public = groq_backend(settings, http) if settings.direct_public_backend == "groq" else local
    return DirectClient(public=public, local=local)
