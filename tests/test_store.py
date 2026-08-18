"""Store behavior: append-only log, catch-up, dedupe, merge — with a fake
embedder so no model is needed."""

import time

import numpy as np
import pytest

from mnema import core
from mnema.store import Entry, Store, merge


class FakeEmbedder:
    """Deterministic hash-based unit embeddings; no torch, no network.
    (hashlib, NOT builtin hash() — that one is randomized per process.)"""
    dim = 32

    def passages(self, texts):
        import hashlib
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:4], "big")
            v = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)

    def queries(self, texts):
        return self.passages(texts)


def make_store(tmp_path, name="s1"):
    return Store.init(tmp_path / name, model="fake", dim=64, beta=1.0,
                      sigma=None, seed=7, d_embed=32)


def test_append_is_deduplicating_and_catch_up_folds(tmp_path):
    s = make_store(tmp_path)
    e = Entry(text="the sky is blue", topic="sky", at="2026-01-01T00:00:00Z")
    assert s.append([e]) == 1
    assert s.append([e]) == 0                            # exact dupe dropped
    st = s.catch_up(FakeEmbedder(), quiet=True)
    assert st.n == 1


def test_cross_session_catch_up_matches_one_shot(tmp_path):
    entries = [Entry(text=f"fact {i}", topic=f"t{i}", at=f"2026-01-0{i+1}T00:00:00Z")
               for i in range(5)]
    one = make_store(tmp_path, "one")
    one.append(entries)
    st_one = one.catch_up(FakeEmbedder(), quiet=True)

    two = make_store(tmp_path, "two")
    for e in entries:                                    # five separate "sessions"
        two = Store(two.path)                            # fresh open each time
        two.append([e])
        st_two = two.catch_up(FakeEmbedder(), quiet=True)
    assert st_two.n == st_one.n == 5
    assert np.abs(st_one.S - st_two.S).max() < 1e-4


def test_supersession_through_the_store(tmp_path):
    s = make_store(tmp_path)
    s.append([Entry(text="policy is A", topic="policy", at="2026-01-01T00:00:00Z")])
    s.catch_up(FakeEmbedder(), quiet=True)
    s.append([Entry(text="policy is B now", topic="policy", at="2026-02-01T00:00:00Z")])
    st = s.catch_up(FakeEmbedder(), quiet=True)

    emb = FakeEmbedder()
    value_lift, key_lift = s._lift_ops()
    k = key_lift(emb.passages(["policy"]))[0]
    V = s.vectors("vec_v.f16").astype(np.float32)
    scores = V @ (st.S @ k)
    assert scores[1] > scores[0]                         # latest wins

    old = s.fold_prefix("2026-01-15T00:00:00Z")          # time travel
    scores_old = V @ (old.S @ k)
    assert scores_old[0] > scores_old[1]                 # past state has old truth


def test_merge_interleaves_and_reuses_vectors(tmp_path):
    a, b = make_store(tmp_path, "a"), make_store(tmp_path, "b")
    a.append([Entry(text="alpha ruling", topic="alpha", at="2026-01-01T00:00:00Z")])
    b.append([Entry(text="beta ruling", topic="beta", at="2026-01-02T00:00:00Z")])
    a.catch_up(FakeEmbedder(), quiet=True)
    b.catch_up(FakeEmbedder(), quiet=True)
    m = merge(a, b, tmp_path / "m")
    assert [e.topic for e in m.entries()] == ["alpha", "beta"]
    assert m.load_state().n == 2


def test_merge_refuses_config_mismatch(tmp_path):
    a = make_store(tmp_path, "a")
    b = Store.init(tmp_path / "b", model="fake", dim=128, beta=1.0,
                   sigma=None, seed=7, d_embed=32)
    with pytest.raises(SystemExit):
        merge(a, b, tmp_path / "m")


class ControlledEmbedder:
    """Preset embeddings so displacement geometry is exact and deterministic."""
    dim = 32

    def __init__(self, table):
        self.table = table

    def passages(self, texts):
        rows = []
        for t in texts:
            v = np.asarray(self.table[t], dtype=np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows)

    def queries(self, texts):
        return self.passages(texts)


def _controlled(tmp_path, name):
    u = np.zeros(32); u[0] = 1.0
    orth = np.zeros(32); orth[1] = 1.0
    near = 0.9 * u + np.sqrt(1 - 0.81) * orth      # cos 0.9 to u in embed space
    far = np.zeros(32); far[2] = 1.0
    table = {"old policy": u, "new policy": near, "unrelated fact": far}
    return make_store(tmp_path, name), ControlledEmbedder(table), table


def test_inferred_displacement_attenuates_old_belief(tmp_path):
    s, emb, _ = _controlled(tmp_path, "inf")
    added, disp = s.remember_inferred(
        Entry(text="old policy", at="2026-01-01T00:00:00Z"), emb)
    assert added and disp == []
    added, disp = s.remember_inferred(
        Entry(text="new policy", at="2026-02-01T00:00:00Z"), emb)
    assert added
    assert [e.text for e, w in disp] == ["old policy"]
    assert all(w < 1.0 for _, w in disp)            # invertible by construction

    added, disp = s.remember_inferred(
        Entry(text="unrelated fact", at="2026-03-01T00:00:00Z"), emb)
    assert disp == []                                # below the floor: untouched

    # control: identical entries, no inference
    c = make_store(tmp_path, "ctl")
    c.append([Entry(text="old policy", at="2026-01-01T00:00:00Z"),
              Entry(text="new policy", at="2026-02-01T00:00:00Z"),
              Entry(text="unrelated fact", at="2026-03-01T00:00:00Z")])
    c.catch_up(emb, quiet=True)

    K = s.vectors("vec_k.f16").astype(np.float32)
    k_old = K[0]
    score_inf = float((s.load_state().S @ k_old) @ s.vectors("vec_v.f16").astype(np.float32)[0])
    score_ctl = float((c.load_state().S @ k_old) @ c.vectors("vec_v.f16").astype(np.float32)[0])
    assert score_inf < score_ctl - 0.05              # old belief measurably attenuated


def test_keep_revokes_displacement_exactly(tmp_path):
    s, emb, _ = _controlled(tmp_path, "keep")
    s.remember_inferred(Entry(text="old policy", at="2026-01-01T00:00:00Z"), emb)
    _, disp = s.remember_inferred(Entry(text="new policy", at="2026-02-01T00:00:00Z"), emb)
    old_h = disp[0][0].h

    restored = s.keep(old_h[:8], emb)
    assert restored is not None and restored.text == "old policy"

    c = make_store(tmp_path, "keepctl")
    c.append([Entry(text="old policy", at="2026-01-01T00:00:00Z"),
              Entry(text="new policy", at="2026-02-01T00:00:00Z")])
    c.catch_up(emb, quiet=True)
    np.testing.assert_allclose(s.load_state().S, c.load_state().S, atol=1e-5)


def test_bulk_ingest_inferred_equals_sequential_remembers(tmp_path):
    entries = [Entry(text="old policy", at="2026-01-01T00:00:00Z"),
               Entry(text="new policy", at="2026-02-01T00:00:00Z"),
               Entry(text="unrelated fact", at="2026-03-01T00:00:00Z")]

    seq, emb, _ = _controlled(tmp_path, "seq")
    for e in entries:
        seq.remember_inferred(Entry(text=e.text, at=e.at), emb)

    bulk, emb2, _ = _controlled(tmp_path, "bulk")
    added, n_disp = bulk.ingest_inferred(
        [Entry(text=e.text, at=e.at) for e in entries], emb2)

    assert added == 3 and n_disp == 1
    assert [e.displaces for e in bulk.entries()] == [e.displaces for e in seq.entries()]
    np.testing.assert_allclose(bulk.load_state().S, seq.load_state().S, atol=1e-5)


def test_bulk_ingest_inferred_scales_subquadratically(tmp_path):
    """Regression guard: displacement inference on a bulk load compares each
    new key against every strictly-prior key — inherent (any prior belief
    can be displaced), same candidate pool a sequential `remember` would see.
    But doing that comparison as one small matvec + full argsort per entry
    is scalar-loop overhead on top of the inherent cost, and it compounds
    badly at scale. At a production-scale fold dimension, 8000 keyless
    entries must stay well clear of what a per-row Python loop takes."""
    dim = 1024
    s = Store.init(tmp_path / "scale", model="fake", dim=dim, beta=1.0,
                   sigma=None, seed=7, d_embed=32)
    n = 8000
    entries = [Entry(text=f"synthetic fact {i}",
                     at=f"2026-01-{1 + i // 1000:02d}T00:00:{i % 60:02d}.{i % 1000:03d}Z")
              for i in range(n)]
    t0 = time.perf_counter()
    added, _ = s.ingest_inferred(entries, FakeEmbedder())
    elapsed = time.perf_counter() - t0
    assert added == n
    # blocked-matmul + argpartition finishes in ~1.5-2s here; a per-row
    # matvec+argsort loop takes 6s+ at this N/D — generous middle ground.
    assert elapsed < 4.0


def test_forget_exactly_reverts_an_accidental_ingest(tmp_path):
    s = make_store(tmp_path, "cooked")
    s.append([Entry(text="carefully cooked belief", topic="craft",
                    at="2026-01-01T00:00:00Z")])
    cooked = s.catch_up(FakeEmbedder(), quiet=True).S.copy()

    accident = [Entry(text=f"accidental bulk fact {i}", topic=f"acc{i}",
                      at=f"2026-02-0{i+1}T00:00:00Z") for i in range(3)]
    s.append(accident)
    s.catch_up(FakeEmbedder(), quiet=True)
    assert not np.allclose(s.load_state().S, cooked)             # damage is real

    forgotten = s.forget([e.h for e in accident])
    assert len(forgotten) == 3
    np.testing.assert_allclose(s.load_state().S, cooked, atol=1e-5)   # exact undo

    # forgotten entries are gone from retrieval and from latest-by-topic
    assert "acc0" not in s.latest_by_topic()
    # and the log still records everything: 1 + 3 + 3 retraction events
    assert len(s.entries()) == 7


def test_cli_remember_renders_displaced_blocks_with_full_text_and_reading(
        tmp_path, monkeypatch, capsys):
    import argparse
    from mnema import cli
    s, emb, _ = _controlled(tmp_path, "render")
    monkeypatch.setattr("mnema.store.Embedder", lambda model: emb)
    s.remember_inferred(Entry(text="old policy", at="2026-01-01T00:00:00Z"), emb)
    old_h = s.entries()[0].h

    cli.cmd_remember(argparse.Namespace(
        store=str(s.path), text="new policy", topic=None, slot=None, question=None))
    out = capsys.readouterr().out
    disp = next(w for x in s.entries() for th, w in x.displaces if th == old_h)
    open_tag = next(l for l in out.splitlines() if l.startswith("<displaced "))
    assert f'h="{old_h[:8]}"' in open_tag and 'at="2026-01-01"' in open_tag
    assert f'attenuated="{disp:.2f}"' in open_tag
    assert f'reading="{cli._attenuation_reading(disp)}"' in open_tag
    assert "\n  old policy\n</displaced>" in out           # full text, then close
    assert "mnema keep <hash>" in out
