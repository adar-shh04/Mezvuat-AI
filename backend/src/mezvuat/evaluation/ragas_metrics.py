"""RAGAS 0.4 collections metrics behind Mezvuat's judge (ADR-0012). Needs the `eval` dependency group.

Metrics: `Faithfulness`, `AnswerRelevancy` (judge + embeddings), `ContextPrecisionWithReference`, `ContextRecall`.
The RAGAS LLM is a thin adapter over a `Judge`, so the judge's pinning, data policy and failure rules apply to every
RAGAS call. Embeddings for answer relevancy are Mezvuat's own (bge-small by default), through `embed`.

A metric whose inputs are missing (no reference answer, no retrieved contexts, an abstained answer) is skipped, not
scored. Inference errors from the judge propagate: the evaluation run fails rather than switching judges.
"""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from mezvuat.evaluation.judge import Judge

# RAGAS sends anonymous usage events to its vendor unless told not to. Nothing leaves the process for analytics.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def ragas_version() -> str:
    from importlib.metadata import version

    return version("ragas")


def _llm(judge: Judge) -> Any:
    from ragas.llms.base import InstructorBaseRagasLLM

    class JudgeLLM(InstructorBaseRagasLLM):
        def generate(self, prompt: str, response_model: Any) -> Any:
            return judge.structured(prompt, response_model)

        async def agenerate(self, prompt: str, response_model: Any) -> Any:
            return await asyncio.to_thread(judge.structured, prompt, response_model)

    return JudgeLLM()


def _embeddings(embed: Callable[[str], list[float]]) -> Any:
    from ragas.embeddings.base import BaseRagasEmbedding

    class Embeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return embed(text)

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return await asyncio.to_thread(embed, text)

    return Embeddings()


@dataclass
class RagasScorer:
    judge: Judge
    embed: Callable[[str], list[float]]
    # Questions generated per answer for answer relevancy. RAGAS defaults to 3; each is one judge call.
    strictness: int = 3

    def __post_init__(self) -> None:
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
        )

        llm = _llm(self.judge)
        self._faithfulness = Faithfulness(llm=llm)
        self._relevancy = AnswerRelevancy(llm=llm, embeddings=_embeddings(self.embed), strictness=self.strictness)
        self._precision = ContextPrecisionWithReference(llm=llm)
        self._recall = ContextRecall(llm=llm)

    async def _score(
        self, question: str, answer: str | None, contexts: Sequence[str], reference: str | None
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        ctx = list(contexts)
        if answer and ctx:
            out["faithfulness"] = (
                await self._faithfulness.ascore(user_input=question, response=answer, retrieved_contexts=ctx)
            ).value
        if answer:
            out["answer_relevancy"] = (await self._relevancy.ascore(user_input=question, response=answer)).value
        if reference and ctx:
            out["context_precision"] = (
                await self._precision.ascore(user_input=question, reference=reference, retrieved_contexts=ctx)
            ).value
            out["context_recall"] = (
                await self._recall.ascore(user_input=question, retrieved_contexts=ctx, reference=reference)
            ).value
        # RAGAS returns NaN when it cannot score (e.g. no statements); that is "not scored", not 0.
        return {k: float(v) for k, v in out.items() if v is not None and not math.isnan(float(v))}

    def score(
        self, question: str, answer: str | None, contexts: Sequence[str], reference: str | None
    ) -> dict[str, float]:
        return asyncio.run(self._score(question, answer, contexts, reference))
