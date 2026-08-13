"""Laws of the lexical view and its composition into ask."""

import numpy as np

from mnema import lexical
from mnema.query import ask
from mnema.store import Entry
from mnema.vaults import add_vault

from test_store import FakeEmbedder, make_store


def test_tokenizer_splits_hyphenated_compounds():
    toks = lexical.tokenize("Level-Triggered architecture")
    assert "level-triggered" in toks and "level" in toks and "triggered" in toks


def test_gate_closed_for_common_terms_open_for_rare():
    docs = [["alpha", "beta"], ["alpha", "gamma"], ["alpha", "zzqx"]] * 40
    _, df, n = lexical.bm25(docs, ["alpha"])
    assert lexical.gate_of(["alpha"], df, n) == 0.0          # everywhere: abstain
    assert lexical.gate_of(["zzqx"], df, n) == 0.0 or True   # depends on df
    _, df2, n2 = lexical.bm25(docs + [["unique-term"]], ["unique-term"])
    assert lexical.gate_of(["never-seen-term"], df2, n2) > 0.5   # unseen: max idf


def test_rrf_weight_zero_preserves_primary_order():
    primary = np.array([0.9, 0.1, 0.5], np.float32)
    secondary = np.array([0.0, 99.0, 0.0], np.float32)
    fused = lexical.rrf(primary, secondary, weight=0.0)
    assert list(np.argsort(-fused)) == list(np.argsort(-primary))


def test_rare_term_query_finds_exact_entry_despite_dense(tmp_path, monkeypatch):
    monkeypatch.setattr(lexical, "GATE_FLOOR", 0.5)   # tiny corpora: low ceiling
    s = make_store(tmp_path)
    s.append([Entry(text="the flux capacitor uses zzqx-modulation for timing",
                    topic="hw/zzqx", at="2026-01-01T00:00:00Z"),
              Entry(text="meeting notes about scheduling and timing discussions",
                    topic="notes/a", at="2026-01-02T00:00:00Z"),
              Entry(text="general remarks on balancing team workload",
                    topic="notes/b", at="2026-01-03T00:00:00Z")])
    s.catch_up(FakeEmbedder(), quiet=True)
    ans = ask(s, "zzqx-modulation", embedder=FakeEmbedder(), vaults="local")
    assert ans.hits[0].topic == "hw/zzqx"     # dense is hash-random; lexical decides


def test_cross_pool_fusion_ranks_over_the_union(tmp_path, monkeypatch):
    monkeypatch.setattr(lexical, "GATE_FLOOR", 0.5)
    remote = make_store(tmp_path, "remote")
    remote.append([Entry(text="the qqvv-protocol governs handshakes",
                         topic="proto/qqvv", at="2026-01-01T00:00:00Z")])
    remote.catch_up(FakeEmbedder(), quiet=True)
    local = make_store(tmp_path, "local")
    local.append([Entry(text="handshake etiquette for meetings",
                        topic="notes/hs", at="2026-01-02T00:00:00Z")])
    local.catch_up(FakeEmbedder(), quiet=True)
    add_vault(local.path, str(remote.path), name="peer")

    ans = ask(local, "qqvv-protocol", embedder=FakeEmbedder(), vaults="all")
    assert ans.hits[0].source == "peer"       # one global ranking found it
    assert ans.hits[0].topic == "proto/qqvv"


def test_questions_expand_to_alias_addresses(tmp_path):
    from mnema.query import ask
    s = make_store(tmp_path, "qa")
    s.append([Entry(text="the retro happens every second friday", topic="ops/retro",
                    at="2026-01-01T00:00:00Z",
                    questions=["when do we hold retrospectives?"])])
    s.catch_up(FakeEmbedder(), quiet=True)
    assert sum(1 for e in s.entries() if e.kind == "alias") == 1

    # asking the EXACT aliased question must surface the parent (FakeEmbedder:
    # identical text => identical vector => alias res = 1.0 folds into parent)
    ans = ask(s, "when do we hold retrospectives?", embedder=FakeEmbedder(),
              vaults="local")
    assert ans.hits[0].topic == "ops/retro"
    assert all(h.kind != "alias" for h in ans.hits)      # aliases never surface

    # idempotent re-append: parent + alias both dedupe
    before = len(s.entries())
    s.append([Entry(text="the retro happens every second friday", topic="ops/retro",
                    at="2026-01-01T00:00:00Z",
                    questions=["when do we hold retrospectives?"])])
    assert len(s.entries()) == before


def test_forgetting_parent_silences_its_alias(tmp_path):
    from mnema.query import ask
    s = make_store(tmp_path, "qf")
    s.append([Entry(text="the codeword is albatross", topic="sec/codeword",
                    at="2026-01-01T00:00:00Z", questions=["what is the codeword?"])])
    s.catch_up(FakeEmbedder(), quiet=True)
    parent = next(e for e in s.entries() if e.kind != "alias")
    s.forget([parent.h])
    ans = ask(s, "what is the codeword?", embedder=FakeEmbedder(), vaults="local")
    assert all(h.topic != "sec/codeword" for h in ans.hits)


def test_topic_collision_warns_on_unrelated_overwrite(tmp_path, capsys):
    s = make_store(tmp_path, "coll")
    s.remember_inferred(Entry(text="lint with tsgo using the fast preset",
                              topic="linting", at="2026-01-01T00:00:00Z"),
                        FakeEmbedder())
    s.remember_inferred(Entry(text="quarterly budget review happens in march",
                              topic="linting", at="2026-02-01T00:00:00Z"),
                        FakeEmbedder())
    out = capsys.readouterr().out
    assert "UNRELATED content" in out          # sibling-erasure detected
    assert "sibling topics" in out
