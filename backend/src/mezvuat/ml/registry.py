"""Known models and how to use them. Revisions are pinned so experiments are reproducible (ADR-0006)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbedderSpec:
    model_id: str
    revision: str | None
    dim: int
    max_seq_length: int
    query_prefix: str = ""
    document_prefix: str = ""


@dataclass(frozen=True)
class RerankerSpec:
    model_id: str
    revision: str | None
    max_length: int = 512


_BGE_QUERY = "Represent this sentence for searching relevant passages: "

EMBEDDERS: dict[str, EmbedderSpec] = {
    spec.model_id: spec
    for spec in (
        EmbedderSpec("sentence-transformers/all-MiniLM-L6-v2", "1110a243fdf4706b3f48f1d95db1a4f5529b4d41", 384, 256),
        EmbedderSpec(
            "BAAI/bge-small-en-v1.5",
            "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            384,
            512,
            query_prefix=_BGE_QUERY,
        ),
        EmbedderSpec(
            "BAAI/bge-base-en-v1.5",
            "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
            768,
            512,
            query_prefix=_BGE_QUERY,
        ),
        EmbedderSpec(
            "intfloat/e5-small-v2",
            "ffb93f3bd4047442299a41ebb6fa998a38507c52",
            384,
            512,
            query_prefix="query: ",
            document_prefix="passage: ",
        ),
        EmbedderSpec(
            "Snowflake/snowflake-arctic-embed-s",
            "e596f507467533e48a2e17c007f0e1dacc837b33",
            384,
            512,
            query_prefix=_BGE_QUERY,
        ),
    )
}

RERANKERS: dict[str, RerankerSpec] = {
    spec.model_id: spec
    for spec in (
        RerankerSpec("cross-encoder/ms-marco-MiniLM-L6-v2", "233902d25c440f23af6f7d6e94d2946bac0bee0a"),
        RerankerSpec("BAAI/bge-reranker-base", "2cfc18c9415c912f9d8155881c133215df768a70"),
        RerankerSpec("Alibaba-NLP/gte-reranker-modernbert-base", "f7481e6055501a30fb19d090657df9ec1f79ab2c"),
    )
}


def get_embedder_spec(model_id: str) -> EmbedderSpec:
    try:
        return EMBEDDERS[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown embedding model {model_id!r}; add it to the registry") from exc


def get_reranker_spec(model_id: str) -> RerankerSpec:
    try:
        return RERANKERS[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown reranker model {model_id!r}; add it to the registry") from exc


@dataclass(frozen=True)
class NliSpec:
    """An NLI cross-encoder. `labels` is the model's output order (its `config.json` `id2label`)."""

    model_id: str
    revision: str
    labels: tuple[str, str, str]
    max_length: int = 512
    licence: str = ""


NLI_MODELS: dict[str, NliSpec] = {
    spec.model_id: spec
    for spec in (
        # Apache-2.0; base model microsoft/deberta-v3-small is MIT. ADR-0009 amendment of 2026-11-16.
        NliSpec(
            "cross-encoder/nli-deberta-v3-small",
            "fa2804872c3b4bd748f38c0185cc85775361e735",
            ("contradiction", "entailment", "neutral"),
            licence="Apache-2.0",
        ),
    )
}


def get_nli_spec(model_id: str) -> NliSpec:
    try:
        return NLI_MODELS[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown NLI model {model_id!r}; add it to the registry") from exc
