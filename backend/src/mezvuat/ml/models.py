"""sentence-transformers implementations of the embedder and reranker interfaces.

Inference uses fp16 on CUDA and fp32 on CPU. torch is imported lazily so importing this module stays cheap.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from mezvuat.ml.registry import EmbedderSpec, RerankerSpec, get_embedder_spec, get_reranker_spec


def resolve_device(setting: str = "auto") -> str:
    import torch

    if setting == "cpu":
        return "cpu"
    if setting == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("ML_DEVICE=cuda but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _dtype_kwargs(device: str) -> dict[str, Any]:
    import torch

    return {"torch_dtype": torch.float16} if device == "cuda" else {}


class SentenceTransformerEmbedder:
    def __init__(self, spec: EmbedderSpec, *, device: str = "auto", batch_size: int = 64) -> None:
        from sentence_transformers import SentenceTransformer

        self.spec = spec
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            spec.model_id,
            revision=spec.revision,
            device=self.device,
            model_kwargs=_dtype_kwargs(self.device),
        )
        self._model.max_seq_length = spec.max_seq_length

    @classmethod
    def from_model_id(cls, model_id: str, *, device: str = "auto", batch_size: int = 64) -> SentenceTransformerEmbedder:
        return cls(get_embedder_spec(model_id), device=device, batch_size=batch_size)

    @property
    def dim(self) -> int:
        return self.spec.dim

    def _encode(self, texts: list[str], prefix: str) -> npt.NDArray[np.float32]:
        prefixed = [prefix + text for text in texts] if prefix else texts
        encoded = self._model.encode(
            prefixed,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded, dtype=np.float32)

    def embed_documents(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return self._encode(texts, self.spec.document_prefix)

    def embed_queries(self, texts: list[str]) -> npt.NDArray[np.float32]:
        return self._encode(texts, self.spec.query_prefix)

    def embed_query(self, query: str) -> list[float]:
        return [float(x) for x in self.embed_queries([query])[0]]


class CrossEncoderReranker:
    def __init__(self, spec: RerankerSpec, *, device: str = "auto", batch_size: int = 32) -> None:
        from sentence_transformers import CrossEncoder

        self.spec = spec
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = CrossEncoder(
            spec.model_id,
            revision=spec.revision,
            device=self.device,
            max_length=spec.max_length,
            model_kwargs=_dtype_kwargs(self.device),
            trust_remote_code=False,
        )

    @classmethod
    def from_model_id(cls, model_id: str, *, device: str = "auto", batch_size: int = 32) -> CrossEncoderReranker:
        return cls(get_reranker_spec(model_id), device=device, batch_size=batch_size)

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(s) for s in np.asarray(scores, dtype=np.float32)]


class HfTokenCounter:
    """Counts tokens with the embedder's own tokenizer and splits long text by token windows."""

    def __init__(self, model_id: str, revision: str | None = None) -> None:
        from transformers import AutoTokenizer

        self._tokenizer: Any = AutoTokenizer.from_pretrained(model_id, revision=revision)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def split(self, text: str, max_tokens: int) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = self._tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)[
            "offset_mapping"
        ]
        return [
            (offsets[i][0], offsets[min(i + max_tokens, len(offsets)) - 1][1])
            for i in range(0, len(offsets), max_tokens)
        ]
