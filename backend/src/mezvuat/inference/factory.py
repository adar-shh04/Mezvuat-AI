"""Builds the configured `LLMClient`. Only this module turns settings into provider endpoints."""

from __future__ import annotations

import httpx
from pydantic import SecretStr

from mezvuat.core.config import Settings
from mezvuat.inference.direct import DirectClient, OpenAICompatibleBackend
from mezvuat.inference.errors import InferenceRejectedError
from mezvuat.inference.modelmesh import (
    CircuitBreaker,
    ContractNegotiator,
    ModelMeshBackend,
    ModelMeshClient,
    ModelMeshOps,
)
from mezvuat.inference.types import LLMClient

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


def direct_client(settings: Settings, http: httpx.Client | None = None) -> DirectClient:
    local = ollama_backend(settings, http)
    public = groq_backend(settings, http) if settings.direct_public_backend == "groq" else local
    return DirectClient(public=public, local=local)


def _modelmesh_url(settings: Settings) -> str:
    url = settings.modelmesh_url
    if url is None:
        raise InferenceRejectedError(
            f"INFERENCE_MODE={settings.inference_mode} needs MODELMESH_BASE_URL", code="inference_mode_misconfigured"
        )
    return url


def _secret(value: SecretStr | None) -> str | None:
    return value.get_secret_value() if value is not None and value.get_secret_value() else None


def modelmesh_ops(settings: Settings, http: httpx.Client | None = None) -> ModelMeshOps | None:
    """Feedback, deletion and cache purges. None when ModelMesh is not configured at all."""
    if settings.modelmesh_url is None:
        return None
    return ModelMeshOps(
        base_url=settings.modelmesh_url,
        api_key=_secret(settings.modelmesh_api_key),
        http=http or httpx.Client(timeout=30.0),
    )


def namespace_secret(settings: Settings) -> bytes:
    secret = _secret(settings.mezvuat_secret_key)
    if secret is None:
        raise InferenceRejectedError(
            "ModelMesh needs MEZVUAT_SECRET_KEY to derive cache namespaces", code="inference_mode_misconfigured"
        )
    return secret.encode("utf-8")


def modelmesh_client(settings: Settings, http: httpx.Client | None = None) -> ModelMeshClient:
    url = _modelmesh_url(settings)
    client = http or httpx.Client(timeout=settings.modelmesh_timeout_s)
    api_key = _secret(settings.modelmesh_api_key)
    negotiator = ContractNegotiator(
        base_url=url,
        http=client,
        api_key=api_key,
        contract_ttl_s=settings.modelmesh_contract_ttl_s,
        calibration_ttl_s=settings.modelmesh_calibration_ttl_s,
    )
    backend = ModelMeshBackend(
        base_url=url,
        negotiator=negotiator,
        namespace_secret=namespace_secret(settings),
        api_key=api_key,
        min_quality=settings.modelmesh_min_quality,
        http=client,
    )
    ops = ModelMeshOps(base_url=url, api_key=api_key, http=client)
    breaker = CircuitBreaker(
        probe=ops.health, threshold=settings.modelmesh_breaker_failures, open_s=settings.modelmesh_breaker_open_s
    )
    fallback = direct_client(settings, http) if settings.inference_mode == "modelmesh_with_fallback" else None
    return ModelMeshClient(backend=backend, breaker=breaker, fallback=fallback)


def build_llm_client(settings: Settings, http: httpx.Client | None = None) -> LLMClient:
    if settings.inference_mode == "direct":
        return direct_client(settings, http)
    return modelmesh_client(settings, http)
