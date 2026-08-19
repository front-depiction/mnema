"""The library-scale query path. The law: every optimization is EXACT or
GATED — blocked memmap math returns the same numbers as materialized math;
the prefilter engages only above PRE_MIN rows and must reproduce the exact
path's hits; the entries-free fast path serves only fully caught-up stores
and must match the parsing path; anything else falls back to today's code."""

import numpy as np

from mnema import query as query_mod
from mnema import store as store_mod
from mnema.query import ask
from mnema.store import Entry, Store
from test_store import FakeEmbedder, make_store


def _corpus(n_docs=6, per=4):
    """Small clustered corpus: n_docs 'papers' x per chunks, distinct topics."""
    out = []
    for d in range(n_docs):
        for c in range(per):
            out.append(Entry(
                text=f"paper {d} section {c}: " + " ".join(
                    [f"term{d}"] * 3 + [f"filler{c}", "shared", "vocabulary"]),
                topic=f"lib/doc{d}.pdf#s{c}",
                at=f"2026-01-{d + 1:02d}T00:00:{c:02d}Z", kind="doc"))
    return out


def _hit_keys(ans):
    return [(h.source, h.h) for h in ans.hits]


def test_prefilter_matches_exact_path(tmp_path, monkeypatch):
    s = make_store(tmp_path)
    s.append(_corpus())
    s.catch_up(FakeEmbedder(), quiet=True)

    exact = ask(s, "paper 3 section 2 term3", embedder=FakeEmbedder(), vaults="local")

    monkeypatch.setattr(query_mod, "PRE_MIN", 1)     # engage on a tiny store
    monkeypatch.setattr(query_mod, "PRE_CAND", 10)
    s2 = Store(s.path)
    s2.catch_up(FakeEmbedder(), quiet=True)          # backfills the sidecar
    assert s2._rows_on_disk(store_mod.PRE_K, store_mod.PRE_DIM) == len(s2.entries())
    pre = ask(s2, "paper 3 section 2 term3", embedder=FakeEmbedder(), vaults="local")

    assert _hit_keys(pre) == _hit_keys(exact)
    assert pre.verdict == exact.verdict
    for a, b in zip(pre.hits, exact.hits):
        assert abs(a.score - b.score) < 1e-4          # rescore is EXACT


def test_prefilter_sidecar_appends_with_new_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(query_mod, "PRE_MIN", 1)
    s = make_store(tmp_path)
    s.append(_corpus(n_docs=2))
    s.catch_up(FakeEmbedder(), quiet=True)
    n1 = s._rows_on_disk(store_mod.PRE_K, store_mod.PRE_DIM)
    s.append([Entry(text="late arrival shared vocabulary", topic="lib/late#s0",
                    at="2026-02-01T00:00:00Z", kind="doc")])
    s.catch_up(FakeEmbedder(), quiet=True)
    assert s._rows_on_disk(store_mod.PRE_K, store_mod.PRE_DIM) == n1 + 1 == len(s.entries())


def test_lexical_jurisdiction_survives_prefilter(tmp_path, monkeypatch):
    """A row relevant ONLY lexically (rare jargon, dense-dissimilar) must
    still surface when the prefilter is engaged: the candidate set is dense
    top-k UNION rows posting the query's rare terms."""
    import numpy as np
    from mnema import lexical
    from test_store import ControlledEmbedder
    monkeypatch.setattr(lexical, "GATE_FLOOR", 0.5)
    monkeypatch.setattr(query_mod, "PRE_MIN", 1)
    monkeypatch.setattr(query_mod, "PRE_CAND", 2)

    a = np.zeros(32); a[0] = 1.0
    ortho = np.zeros(32); ortho[1] = 1.0
    table = {"what is the zzqx protocol": a}
    docs = []
    for i in range(6):                       # dense-similar decoys fill top-2
        text = f"routing decoy number {i} about protocols"
        v = a.copy(); v[2 + i] = 0.4
        table[text] = v
        docs.append(Entry(text=text, topic=f"d/{i}", at=f"2026-01-0{i + 1}T00:00:00Z"))
    target = Entry(text="the zzqx handshake pins the session key",
                   topic="d/zzqx", at="2026-01-07T00:00:00Z")
    table[target.text] = ortho               # NO dense similarity to the query
    for e in docs + [target]:
        table[e.topic] = table[e.text]       # key text = topic
    table["d/zzqx"] = ortho

    for e in docs:
        table[e.topic] = table[e.text]
    s = make_store(tmp_path)
    s.append(docs + [target])
    emb = ControlledEmbedder(table)
    s.catch_up(emb, quiet=True)
    ans = ask(Store(s.path), "what is the zzqx protocol", embedder=emb,
              vaults="local")
    assert any(h.topic == "d/zzqx" for h in ans.hits)


def test_fast_path_matches_parsing_path(tmp_path, monkeypatch):
    """A fully caught-up store answers without parsing the whole log."""
    s = make_store(tmp_path)
    s.append(_corpus())
    s.catch_up(FakeEmbedder(), quiet=True)
    baseline = ask(s, "paper 1 section 3 term1", embedder=FakeEmbedder(),
                   vaults="local")

    s2 = Store(s.path)
    calls = {"n": 0}
    orig = Store.entries
    def counting(self):
        calls["n"] += 1
        return orig(self)
    monkeypatch.setattr(Store, "entries", counting)
    fast = ask(s2, "paper 1 section 3 term1", embedder=FakeEmbedder(),
               vaults="local")
    assert calls["n"] == 0                            # no full parse
    assert _hit_keys(fast) == _hit_keys(baseline)
    assert fast.verdict == baseline.verdict
    assert [h.text for h in fast.hits] == [h.text for h in baseline.hits]


def test_fast_path_declines_when_log_leads_derived(tmp_path):
    """New unembedded lines mean the store is NOT current: the ask must take
    the full path (which embeds and folds them) — never serve stale."""
    s = make_store(tmp_path)
    s.append(_corpus(n_docs=2))
    s.catch_up(FakeEmbedder(), quiet=True)
    s.append([Entry(text="brand new fact shared vocabulary", topic="lib/new#s0",
                    at="2026-03-01T00:00:00Z", kind="doc")])
    # FakeEmbedder: identical text embeds identically, so the exact question
    # must surface the unembedded entry — proving the ask embedded it first
    ans = ask(Store(s.path), "brand new fact shared vocabulary",
              embedder=FakeEmbedder(), vaults="local")
    assert any("brand new fact" in h.text for h in ans.hits)


def test_fast_path_declines_for_as_of_and_current(tmp_path):
    s = make_store(tmp_path)
    s.append([Entry(text="policy is A shared", topic="policy",
                    at="2026-01-01T00:00:00Z"),
              Entry(text="policy is B now shared", topic="policy",
                    at="2026-02-01T00:00:00Z")])
    s.catch_up(FakeEmbedder(), quiet=True)
    old = ask(Store(s.path), "policy", embedder=FakeEmbedder(), vaults="local",
              as_of="2026-01-15T00:00:00Z")
    assert any("policy is A" in h.text for h in old.hits)
    assert not any("policy is B" in h.text for h in old.hits)
    cur = ask(Store(s.path), "policy", embedder=FakeEmbedder(), vaults="local",
              current_only=True)
    assert not any("policy is A" in h.text for h in cur.hits)


def test_line_offsets_sidecar_tracks_appends(tmp_path):
    s = make_store(tmp_path)
    s.append(_corpus(n_docs=2))
    off1 = s.line_offsets()
    assert len(off1) == 8 == s.entry_count()
    s.append([Entry(text="tail entry", topic="t", at="2026-04-01T00:00:00Z")])
    off2 = s.line_offsets()
    assert len(off2) == 9
    assert np.array_equal(off1, off2[:8])             # append-only: prefix stable
    e = s.read_rows([8])[0]
    assert e.text == "tail entry" and e.topic == "t"


def test_relate_hop_survives_fast_path(tmp_path, monkeypatch):
    """Cross-store relations still appear when pools are built entries-free."""
    remote = make_store(tmp_path, "remote")
    remote.append([Entry(text="capability membranes attenuate authority",
                         topic="ocap/membranes", at="2026-01-01T00:00:00Z")])
    remote.catch_up(FakeEmbedder(), quiet=True)
    local = make_store(tmp_path, "local")
    local.append([Entry(text="capability membranes attenuate authority",
                        topic="notes/membranes", at="2026-01-02T00:00:00Z")])
    local.catch_up(FakeEmbedder(), quiet=True)
    from mnema import vaults as V
    V.add_vault(local.path, str(remote.path), "remote")
    ans = ask(Store(local.path), "capability membranes attenuate authority",
              embedder=FakeEmbedder(), vaults="all")
    rel = [r for rs in ans.related_by_hit for r in rs]
    assert any(r.source == "remote" for r in rel) or \
           any(h.source == "remote" for h in ans.hits)
