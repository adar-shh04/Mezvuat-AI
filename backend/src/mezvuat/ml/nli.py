"""The NLI cross-encoder behind `mezvuat.citations.support.NliModel` (ADR-0009 amendment, 2026-11-16).

It runs on CPU by default (`NLI_DEVICE=cpu`): a research run scores a handful of claims, so the GPU is not needed,
and it stays free for the embedder and for other local work. torch is imported lazily.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from mezvuat.citations.support import NliScores
from mezvuat.ml.models import resolve_device
from mezvuat.ml.registry import NliSpec, get_nli_spec


class CrossEncoderNli:
    def __init__(self, spec: NliSpec, *, device: str = "cpu", batch_size: int = 16) -> None:
        from sentence_transformers import CrossEncoder

        self.spec = spec
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self._model = CrossEncoder(
            spec.model_id,
            revision=spec.revision,
            device=self.device,
            max_length=spec.max_length,
            trust_remote_code=False,
        )
        config = self._model.model.config.id2label
        found = tuple(str(config[i]).lower() for i in range(len(config)))
        if found != spec.labels:
            raise RuntimeError(f"{spec.model_id} labels are {found}, the registry expects {spec.labels}")
        self._index = {label: n for n, label in enumerate(spec.labels)}

    @classmethod
    def from_model_id(cls, model_id: str, *, device: str = "cpu", batch_size: int = 16) -> CrossEncoderNli:
        return cls(get_nli_spec(model_id), device=device, batch_size=batch_size)

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def predict(self, pairs: Sequence[tuple[str, str]]) -> list[NliScores]:
        if not pairs:
            return []
        probs = self._model.predict(
            list(pairs),
            batch_size=self.batch_size,
            apply_softmax=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        rows = np.asarray(probs, dtype=np.float64).reshape(len(pairs), len(self.spec.labels))
        return [
            NliScores(
                entailment=float(row[self._index["entailment"]]),
                contradiction=float(row[self._index["contradiction"]]),
                neutral=float(row[self._index["neutral"]]),
            )
            for row in rows
        ]
