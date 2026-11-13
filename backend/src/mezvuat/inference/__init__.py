"""Inference layer (ADR-0011). The only package that talks HTTP to LLM providers."""

from mezvuat.inference.direct import DirectClient, OpenAICompatibleBackend
from mezvuat.inference.errors import (
    FATAL_INFERENCE_CODES,
    InferenceError,
    InferenceNotPermittedError,
    InferenceRateLimitedError,
    InferenceRejectedError,
    InferenceUnavailableError,
    StreamIncompleteError,
    StructuredOutputInvalidError,
)
from mezvuat.inference.factory import build_llm_client
from mezvuat.inference.schema import portability_problems, portable_schema
from mezvuat.inference.types import (
    TASK_TYPES,
    ChatMessage,
    Correlation,
    DataPolicy,
    LLMAttempt,
    LLMClient,
    LLMRequest,
    LLMResult,
    LLMStreamEvent,
    LLMUsage,
    StreamDelta,
    StreamDone,
    TaskType,
)

__all__ = [
    "FATAL_INFERENCE_CODES",
    "TASK_TYPES",
    "ChatMessage",
    "Correlation",
    "DataPolicy",
    "DirectClient",
    "InferenceError",
    "InferenceNotPermittedError",
    "InferenceRateLimitedError",
    "InferenceRejectedError",
    "InferenceUnavailableError",
    "LLMAttempt",
    "LLMClient",
    "LLMRequest",
    "LLMResult",
    "LLMStreamEvent",
    "LLMUsage",
    "OpenAICompatibleBackend",
    "StreamDelta",
    "StreamDone",
    "StreamIncompleteError",
    "StructuredOutputInvalidError",
    "TaskType",
    "build_llm_client",
    "portability_problems",
    "portable_schema",
]
