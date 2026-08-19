"""Embedding economy: the forward pass is >95% of ingest cost, so the store
must never run it twice for the same text. One fused pass per block computes
values AND keys (keyless entries share the text: embed once, lift twice);
blocks checkpoint to disk so huge ingests are resumable and memory-flat; the
content-addressed cache makes repeat texts free across commands and stores."""

import numpy as np

from mnema import store as store_mod
from mnema.store import Entry, Store, recompile
from test_store import FakeEmbedder, make_store


class CountingEmbedder(FakeEmbedder):
    """Records every text that pays a forward pass."""

    def __init__(self):
        self.embedded: list[str] = []

    def passages(self, texts):
        self.embedded.extend(texts)
        return super().passages(texts)


def test_keyless_ingest_embeds_each_unique_text_once(tmp_path):
    s = make_store(tmp_path)
    s.append([Entry(text=f"keyless belief {i}", at=f"2026-01-01T00:00:0{i}Z")
              for i in range(5)])
    emb = CountingEmbedder()
    s.catch_up(emb, quiet=True)
    # keyless: key text == value text — one forward pass each, not two
    assert sorted(emb.embedded) == sorted(f"keyless belief {i}" for i in range(5))


def test_topical_ingest_embeds_text_and_topic_once_each(tmp_path):
    s = make_store(tmp_path)
    s.append([Entry(text=f"chunk {i}", topic=f"paper.pdf#s{i}",
                    at=f"2026-01-01T00:00:0{i}Z") for i in range(3)])
    emb = CountingEmbedder()
    s.catch_up(emb, quiet=True)
    want = [f"chunk {i}" for i in range(3)] + [f"paper.pdf#s{i}" for i in range(3)]
    assert sorted(emb.embedded) == sorted(want)


def test_alias_pays_one_pass_for_its_question_only(tmp_path):
    s = make_store(tmp_path)
    e = Entry(text="retros are fridays", topic="ops/retro",
              at="2026-01-01T00:00:00Z", questions=["when do we hold retros?"])
    s.append([e])
    emb = CountingEmbedder()
    s.catch_up(emb, quiet=True)
    # alias value is COPIED from the parent row — its text must not be
    # embedded a second time for the value side
    assert emb.embedded.count("when do we hold retros?") == 1


def test_blocked_embedding_matches_one_shot_exactly(tmp_path, monkeypatch):
    entries = [Entry(text=f"fact {i}", topic=(f"t{i}" if i % 2 else None),
                     at=f"2026-01-0{i + 1}T00:00:00Z") for i in range(7)]
    one = make_store(tmp_path, "one")
    one.append(entries)
    st_one = one.catch_up(FakeEmbedder(), quiet=True)

    monkeypatch.setattr(store_mod, "EMBED_BLOCK", 2)
    blk = make_store(tmp_path, "blk")
    blk.append(entries)
    st_blk = blk.catch_up(FakeEmbedder(), quiet=True)

    assert st_blk.n == st_one.n == 7
    assert np.array_equal(np.asarray(one.vectors("vec_v.f16")),
                          np.asarray(blk.vectors("vec_v.f16")))
    assert np.array_equal(np.asarray(one.vectors("vec_k.f16")),
                          np.asarray(blk.vectors("vec_k.f16")))
    assert np.abs(st_one.S - st_blk.S).max() < 1e-5


def test_blocked_embedding_is_resumable_mid_ingest(tmp_path, monkeypatch):
    """A crash between blocks loses nothing: rerun continues from disk."""
    monkeypatch.setattr(store_mod, "EMBED_BLOCK", 2)
    entries = [Entry(text=f"fact {i}", at=f"2026-01-0{i + 1}T00:00:00Z")
               for i in range(5)]
    s = make_store(tmp_path)
    s.append(entries)

    calls = {"n": 0}
    class DiesAfterTwoBlocks(FakeEmbedder):
        def passages(self, texts):
            calls["n"] += 1
            if calls["n"] > 2:
                raise RuntimeError("simulated crash")
            return super().passages(texts)

    try:
        s.catch_up(DiesAfterTwoBlocks(), quiet=True)
    except RuntimeError:
        pass
    survived = s._rows_on_disk("vec_v.f16")
    assert survived == 4                                # two blocks checkpointed

    emb = CountingEmbedder()
    st = s.catch_up(emb, quiet=True)                    # resume, don't restart
    assert st.n == 5
    assert emb.embedded == ["fact 4"]                   # only the missing tail


def test_torn_append_is_repaired_not_compounded(tmp_path):
    """If a crash leaves vec_v longer than vec_k, the orphan rows must be
    truncated before the next append — otherwise every later row misaligns
    with its log line."""
    s = make_store(tmp_path)
    s.append([Entry(text="a", at="2026-01-01T00:00:00Z")])
    s.catch_up(FakeEmbedder(), quiet=True)
    with (s.path / "vec_v.f16").open("ab") as f:        # orphan row: torn append
        f.write(np.zeros(s.dim, np.float16).tobytes())
    s.append([Entry(text="b", at="2026-01-02T00:00:00Z")])
    s.catch_up(FakeEmbedder(), quiet=True)
    V = s.vectors("vec_v.f16")
    assert V.shape[0] == 2                              # repaired, not 3
    value_lift, _ = s._lift_ops()
    want = value_lift(FakeEmbedder().passages(["b"]))[0]
    assert np.abs(np.asarray(V[1], np.float32) - want).max() < 2e-3


def test_recompile_rebuilds_everything_from_the_log(tmp_path):
    src = make_store(tmp_path, "src")
    src.append([Entry(text="alpha ruling", topic="alpha", at="2026-01-01T00:00:00Z"),
                Entry(text="keyless note", at="2026-01-02T00:00:00Z")])
    src.catch_up(FakeEmbedder(), quiet=True)

    class Fake64(FakeEmbedder):
        dim = 64
    out = recompile(src, tmp_path / "out", model="fake-v2", embedder=Fake64())
    assert [e.h for e in out.entries()] == [e.h for e in src.entries()]
    assert out.cfg["model"] == "fake-v2"
    assert out.cfg["d_embed"] == 64
    for key in ("dim", "beta", "sigma", "seed"):
        assert out.cfg[key] == src.cfg[key]
    assert out.load_state().n == len(out.entries())
