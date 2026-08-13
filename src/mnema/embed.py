"""Embedding model registry. Any sentence-transformers model works; these are
the vetted defaults. Embeddings are always unit-norm float32."""

from __future__ import annotations

import numpy as np

MODELS: dict[str, dict] = {
    "BAAI/bge-large-en-v1.5": {
        "dim": 1024,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "BAAI/bge-small-en-v1.5": {
        "dim": 384,
        "query_prefix": "Represent this sentence for searching relevant passages: ",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {"dim": 384, "query_prefix": ""},
    "BAAI/bge-m3": {"dim": 1024, "query_prefix": ""},   # 8192-token window
    "Alibaba-NLP/gte-large-en-v1.5": {"dim": 1024, "query_prefix": "",
                                      "trust_remote_code": True},  # English, 8192
    "thenlper/gte-large": {"dim": 1024, "query_prefix": ""},
}

DEFAULT_MODEL = "BAAI/bge-large-en-v1.5"


def query_prefix(model_name: str) -> str:
    return MODELS.get(model_name, {}).get("query_prefix", "")


def pick_device() -> str:
    import os
    if os.environ.get("MNEMA_DEVICE"):
        return os.environ["MNEMA_DEVICE"]
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    """Lazy-loaded sentence-transformers wrapper (model load dominates startup;
    construct once per process)."""

    def __init__(self, model_name: str, direct: bool = False):
        self.model_name = model_name
        self.direct = direct       # True inside the daemon: never call yourself
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            trc = MODELS.get(self.model_name, {}).get("trust_remote_code", False)
            try:
                # cached weights: no Hub round trip, no version ping
                self._model = SentenceTransformer(self.model_name, device=pick_device(),
                                                  local_files_only=True,
                                                  trust_remote_code=trc)
            except Exception:
                self._model = SentenceTransformer(self.model_name, device=pick_device(),
                                                  trust_remote_code=trc)
        return self._model

    @property
    def dim(self) -> int:
        known = MODELS.get(self.model_name)
        if known:
            return known["dim"]
        return int(self.model.get_sentence_embedding_dimension())

    def passages(self, texts: list[str]) -> np.ndarray:
        if not self.direct:
            from .serve import try_daemon
            via = try_daemon(self.model_name, "passages", texts)
            if via is not None:
                return via
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        ).astype(np.float32)

    def queries(self, texts: list[str]) -> np.ndarray:
        if not self.direct:
            from .serve import try_daemon
            via = try_daemon(self.model_name, "queries", texts)
            if via is not None:
                return via
        prefix = query_prefix(self.model_name)
        return self.passages([prefix + t for t in texts])
