"""Embedding model registry. Any sentence-transformers model works; these are
the vetted defaults. Embeddings are always unit-norm float32.

The forward pass is >95% of ingest cost, so this module never runs it twice
for one text: passages() dedupes within a call, and the content-addressed
cache — (model, sha256(text)) -> raw embedding — makes repeats free across
commands, stores, and vault publishes. Log-level idempotency can't do that:
it is per-store and keyed on the entry hash (which carries the timestamp)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np

MODELS: dict[str, dict] = {
    # 2024-generation encoder: better BEIR than bge-large at 1.9x the measured
    # throughput. seq_len pins the 512-token horizon — small memories retrieve
    # better (measured), so the 8192-token window is deliberately not used.
    "Alibaba-NLP/gte-modernbert-base": {"dim": 768, "query_prefix": "",
                                        "seq_len": 512},
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

DEFAULT_MODEL = "Alibaba-NLP/gte-modernbert-base"

DAEMON_BATCH = 512   # texts per daemon round trip: caps JSON payloads (~11 MB
                     # at 1024 dims) so bulk ingests stay memory-flat


def query_prefix(model_name: str) -> str:
    return MODELS.get(model_name, {}).get("query_prefix", "")


def pick_device() -> str:
    if os.environ.get("MNEMA_DEVICE"):
        return os.environ["MNEMA_DEVICE"]
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "-", model_name)


def _digest(text: str) -> bytes:
    return hashlib.sha256(text.encode()).digest()[:16]


class EmbedCache:
    """Append-only content-addressed cache, one per model: <slug>.key holds
    16-byte digests, <slug>.f16 the position-aligned fp16 rows. Loading
    truncates to the aligned prefix, so a torn append degrades to a miss —
    never to a wrong vector. Appends serialize under an advisory lock."""

    def __init__(self, model_name: str, dim: int | None = None):
        root = Path(os.environ.get("MNEMA_EMBED_CACHE",
                                   Path.home() / ".cache" / "mnema" / "embcache"))
        root.mkdir(parents=True, exist_ok=True)
        slug = _slug(model_name)
        self.kpath = root / f"{slug}.key"
        self.vpath = root / f"{slug}.f16"
        self.mpath = root / f"{slug}.json"
        if dim is None and self.mpath.exists():
            dim = json.loads(self.mpath.read_text())["dim"]
        self.dim = dim
        self._row: dict[bytes, int] = {}
        self._load()

    def _load(self) -> None:
        if not self.dim or not self.kpath.exists() or not self.vpath.exists():
            return
        keys = self.kpath.read_bytes()
        n = min(len(keys) // 16, self.vpath.stat().st_size // (2 * self.dim))
        self._row = {keys[16 * i:16 * i + 16]: i for i in range(n)}

    def get(self, texts: list[str]) -> tuple[dict[int, np.ndarray], list[int]]:
        """(position -> vector) for hits, positions of misses."""
        if not self._row:
            return {}, list(range(len(texts)))
        rows = np.memmap(self.vpath, dtype=np.float16, mode="r").reshape(-1, self.dim)
        hits: dict[int, np.ndarray] = {}
        misses: list[int] = []
        for i, t in enumerate(texts):
            j = self._row.get(_digest(t))
            if j is None or j >= rows.shape[0]:
                misses.append(i)
            else:
                hits[i] = np.asarray(rows[j], dtype=np.float32)
        return hits, misses

    def put(self, texts: list[str], X: np.ndarray) -> None:
        if not len(texts):
            return
        if self.dim is None:
            self.dim = int(X.shape[1])
        if not self.mpath.exists():
            self.mpath.write_text(json.dumps({"dim": self.dim}))
        lock = (self.kpath.parent / ".lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            self._load()                       # another process may have appended
            n = len(self._row)                 # aligned prefix; truncate any
            for p, width in ((self.kpath, 16), (self.vpath, 2 * self.dim)):
                if p.exists() and p.stat().st_size > n * width:
                    os.truncate(p, n * width)  # torn tail before appending past it
            fresh = [i for i, t in enumerate(texts) if _digest(t) not in self._row]
            if not fresh:
                return
            # rows first, keys second: a torn write leaves an orphan row the
            # loader ignores, never a key pointing at a missing row
            with self.vpath.open("ab") as f:
                f.write(X[fresh].astype(np.float16).tobytes())
            with self.kpath.open("ab") as f:
                f.write(b"".join(_digest(texts[i]) for i in fresh))
            base = len(self._row)
            for n, i in enumerate(fresh):
                self._row[_digest(texts[i])] = base + n
        finally:
            lock.close()


class Embedder:
    """Lazy-loaded sentence-transformers wrapper (model load dominates startup;
    construct once per process)."""

    def __init__(self, model_name: str, direct: bool = False):
        self.model_name = model_name
        self.direct = direct       # True inside the daemon: never call yourself
        self._model = None
        self._cache: EmbedCache | None | bool = None    # False = disabled

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            spec = MODELS.get(self.model_name, {})
            trc = spec.get("trust_remote_code", False)
            try:
                # cached weights: no Hub round trip, no version ping
                self._model = SentenceTransformer(self.model_name, device=pick_device(),
                                                  local_files_only=True,
                                                  trust_remote_code=trc)
            except Exception:
                self._model = SentenceTransformer(self.model_name, device=pick_device(),
                                                  trust_remote_code=trc)
            if spec.get("seq_len"):
                self._model.max_seq_length = spec["seq_len"]
        return self._model

    @property
    def dim(self) -> int:
        known = MODELS.get(self.model_name)
        if known:
            return known["dim"]
        return int(self.model.get_sentence_embedding_dimension())

    def cache(self) -> EmbedCache | None:
        if self._cache is None:
            if os.environ.get("MNEMA_EMBED_CACHE_OFF"):
                self._cache = False
            else:
                self._cache = EmbedCache(self.model_name,
                                         MODELS.get(self.model_name, {}).get("dim"))
        return self._cache or None

    def passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), np.float32)
        order = list(dict.fromkeys(texts))              # one pass per unique text
        E = self._unique(order)
        pos = {t: i for i, t in enumerate(order)}
        return E[[pos[t] for t in texts]]

    def queries(self, texts: list[str]) -> np.ndarray:
        prefix = query_prefix(self.model_name)
        return self.passages([prefix + t for t in texts])

    def _unique(self, texts: list[str]) -> np.ndarray:
        cache = self.cache()
        hits, miss = cache.get(texts) if cache else ({}, list(range(len(texts))))
        if not miss:
            return np.stack([hits[i] for i in range(len(texts))])
        fresh, from_daemon = self._encode([texts[i] for i in miss])
        if cache and not from_daemon:
            cache.put([texts[i] for i in miss], fresh)  # the daemon caches its own
        if not hits:
            return fresh
        out = np.empty((len(texts), fresh.shape[1]), np.float32)
        for i, v in hits.items():
            out[i] = v
        out[miss] = fresh
        return out

    def _encode(self, texts: list[str]) -> tuple[np.ndarray, bool]:
        if not self.direct:
            from .serve import try_daemon
            via = try_daemon(self.model_name, "passages", texts)
            if via is not None:
                return via, True
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        ).astype(np.float32), False
