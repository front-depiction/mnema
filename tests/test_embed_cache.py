"""The content-addressed embedding cache: (model, sha256(text)) -> raw
embedding. Log-level idempotency dedupes exact re-appends per store; the cache
is the cross-store, cross-mtime, cross-command layer — the same paper ingested
into two vault stores, a re-converted file whose text didn't change, and the
remember flow's repeated passes over one text all pay the forward pass once."""

import numpy as np
import pytest

from mnema.embed import EmbedCache, Embedder


@pytest.fixture()
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("MNEMA_EMBED_CACHE", str(tmp_path / "embcache"))
    return tmp_path / "embcache"


def test_roundtrip_and_persistence(cache_dir):
    c = EmbedCache("some/model", dim=8)
    X = np.arange(16, dtype=np.float32).reshape(2, 8)
    c.put(["alpha", "beta"], X)
    hits, misses = c.get(["beta", "gamma", "alpha"])
    assert misses == [1]
    assert np.abs(hits[0] - X[1]).max() < 1e-2          # fp16 storage
    assert np.abs(hits[2] - X[0]).max() < 1e-2

    again = EmbedCache("some/model", dim=8)             # fresh process
    hits, misses = again.get(["alpha"])
    assert misses == [] and np.abs(hits[0] - X[0]).max() < 1e-2


def test_models_do_not_share_entries(cache_dir):
    a = EmbedCache("model/a", dim=4)
    a.put(["text"], np.ones((1, 4), np.float32))
    b = EmbedCache("model/b", dim=4)
    _, misses = b.get(["text"])
    assert misses == [0]


def test_torn_write_degrades_to_miss(cache_dir):
    c = EmbedCache("some/model", dim=4)
    c.put(["kept"], np.ones((1, 4), np.float32))
    with c.vpath.open("ab") as f:                       # row without its key
        f.write(np.zeros(4, np.float16).tobytes())
    c2 = EmbedCache("some/model", dim=4)
    hits, misses = c2.get(["kept"])
    assert misses == []                                 # aligned prefix survives


class FakeModel:
    """Stands in for SentenceTransformer: hash-seeded, counts encodes."""

    def __init__(self, dim=8):
        self.dim = dim
        self.encoded: list[str] = []

    def encode(self, texts, **kw):
        import hashlib
        self.encoded.extend(texts)
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)


def make_embedder(model):
    e = Embedder("fake/counting", direct=True)          # direct: never dial a daemon
    e._model = model
    return e


def test_embedder_dedupes_within_a_call(cache_dir, monkeypatch):
    monkeypatch.setenv("MNEMA_EMBED_CACHE_OFF", "1")    # dedupe alone, no cache
    fm = FakeModel()
    emb = make_embedder(fm)
    X = emb.passages(["a", "b", "a", "a", "b"])
    assert sorted(fm.encoded) == ["a", "b"]
    assert X.shape == (5, 8)
    assert np.array_equal(X[0], X[2]) and np.array_equal(X[1], X[4])


def test_embedder_hits_cache_across_calls_and_instances(cache_dir):
    fm = FakeModel()
    emb = make_embedder(fm)
    first = emb.passages(["alpha", "beta"])
    emb.passages(["beta", "gamma"])
    assert sorted(fm.encoded) == ["alpha", "beta", "gamma"]

    fm2 = FakeModel()
    emb2 = make_embedder(fm2)                           # new command, same cache
    again = emb2.passages(["alpha", "delta"])
    assert fm2.encoded == ["delta"]
    assert np.abs(again[0] - first[0]).max() < 1e-2     # fp16 round-trip
