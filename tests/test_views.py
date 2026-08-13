"""Laws of the read-side views fold: segment-invariance, posting-list BM25
equivalence, and ask() indifference to how the views were obtained."""

import numpy as np
import pytest

from mnema import lexical, views
from mnema import query as Q
from mnema.query import ask
from mnema.store import Entry, Store

from test_store import FakeEmbedder, make_store


def _entry_mix(tmp_path) -> list:
    """A log exercising every event kind: topic supersession, keyless notes,
    displacement edges, question aliases, a keep, a retract — appended through
    the store so alias expansion is the real one."""
    s = make_store(tmp_path, "mix")
    a = Entry(text="quorum needs five nodes", topic="q/quorum", at="2026-01-01T00:00:00Z")
    b = Entry(text="quorum lowered to three nodes", topic="q/quorum", at="2026-01-02T00:00:00Z")
    c = Entry(text="retries use exponential backoff", at="2026-01-03T00:00:00Z")
    d = Entry(text="retries now capped at five attempts", at="2026-01-04T00:00:00Z",
              displaces=[[c.h, 0.8]])
    e = Entry(text="standup is at four pm", topic="ops/standup", at="2026-01-05T00:00:00Z",
              questions=["when is standup?"])
    f = Entry(text="scratch note, to be retracted", at="2026-01-06T00:00:00Z")
    s.append([a, b, c, d, e, f])
    s.append([Entry(text=f"keep {c.h}", kind="keep", target=c.h, at="2026-01-07T00:00:00Z")])
    s.append([Entry(text=f"retract {f.h}", kind="retract", target=f.h, at="2026-01-08T00:00:00Z")])
    s.append([Entry(text="post-revocation belief", topic="q/quorum", at="2026-01-09T00:00:00Z"),
              Entry(text="late keyless displacer", at="2026-01-10T00:00:00Z",
                    displaces=[[a.h, 0.75]])])
    return s.entries()


def _views_equal(x: views.Views, y: views.Views) -> None:
    assert x.n == y.n
    for name in ("post_term", "post_row", "post_tf", "doc_len", "mask"):
        assert np.array_equal(getattr(x, name), getattr(y, name)), name
    assert x.terms == y.terms and x.term_id == y.term_id
    assert x.latest_by_topic == y.latest_by_topic
    assert x.topic_groups == y.topic_groups
    assert [list(p) for p in x.disp_pairs] == [list(p) for p in y.disp_pairs]
    assert [list(p) for p in x.alias_parent] == [list(p) for p in y.alias_parent]
    assert x.displaced == y.displaced
    assert x.retracted == y.retracted and x.revoked == y.revoked


def test_advance_in_segments_equals_of_from_scratch(tmp_path):
    entries = _entry_mix(tmp_path)
    whole = views.of(entries)
    n = len(entries)
    for cuts in ([n], [1, n], [3, 6, n], [2, 5, 7, 9, n],
                 list(range(1, n + 1))):
        acc, start = views.Views.empty(), 0
        for cut in cuts:
            acc = views.advance(acc, entries[:cut], start)
            start = cut
        _views_equal(whole, acc)


def test_views_survive_save_load_roundtrip(tmp_path):
    entries = _entry_mix(tmp_path)
    vw = views.of(entries)
    p = tmp_path / "views.npz"
    views.save(vw, p)
    _views_equal(vw, views.load(p))


def test_posting_list_bm25_equals_corpus_scan():
    rng = np.random.default_rng(11)
    vocab = [f"tok{i}" for i in range(60)]
    docs = []
    for i in range(300):
        length = int(rng.integers(0, 25))
        docs.append([] if length == 0 else rng.choice(vocab, length).tolist())
    for q_tokens in (["tok3", "tok7", "tok3"], ["tok0"], ["never-seen"], [],
                     rng.choice(vocab, 6).tolist() + ["rare-unseen"]):
        ref_scores, ref_df, ref_n = lexical.bm25(docs, q_tokens)
        postings = {w: ([], []) for w in set(q_tokens)}
        for i, d in enumerate(docs):
            for w in set(q_tokens):
                c = d.count(w)
                if c:
                    postings[w][0].append(i)
                    postings[w][1].append(c)
        postings = {w: (np.asarray(r, np.int32), np.asarray(t, np.int32))
                    for w, (r, t) in postings.items()}
        doc_len = np.asarray([len(d) for d in docs], np.int32)
        scores, df, n = lexical.bm25_index(postings, doc_len, q_tokens)
        assert n == ref_n
        np.testing.assert_allclose(scores, ref_scores, atol=1e-6)
        assert all(df.get(w, 0) == ref_df.get(w, 0) for w in q_tokens)
        assert lexical.gate_of(q_tokens, df, n) == lexical.gate_of(q_tokens, ref_df, ref_n)


def _snap(ans):
    return (ans.verdict, ans.support,
            [(h.h, h.source, h.at, h.superseded_by, h.displaced) for h in ans.hits],
            [h.score for h in ans.hits])


def _curated_store(tmp_path) -> Store:
    s = make_store(tmp_path, "eq")
    a = Entry(text="the deploy freeze is on fridays", topic="ops/deploy", at="2026-01-01T00:00:00Z")
    b = Entry(text="deploy freeze lifted, canary gated", topic="ops/deploy", at="2026-02-01T00:00:00Z")
    c = Entry(text="retries use exponential backoff", at="2026-03-01T00:00:00Z")
    d = Entry(text="retries now capped at five attempts", at="2026-04-01T00:00:00Z",
              displaces=[[c.h, 0.8]])
    e = Entry(text="standup is at four pm", topic="ops/standup", at="2026-05-01T00:00:00Z",
              questions=["when is standup?"])
    f = Entry(text="scratch note, an accident", at="2026-06-01T00:00:00Z")
    s.append([a, b, c, d, e, f])
    s.catch_up(FakeEmbedder(), quiet=True)
    s.forget([f.h])
    s.keep(c.h[:8])
    return s


QUESTIONS = ["can we deploy on friday?", "when is standup?",
             "exponential backoff retries", "something entirely unwritten"]


def test_ask_equivalent_across_sidecar_load_rebuild_and_rederivation(tmp_path, monkeypatch):
    s = _curated_store(tmp_path)
    assert (s.path / "views.npz").exists()

    def answers():
        fresh = Store(s.path)                     # cold: no in-process views
        out = [_snap(ask(fresh, q, embedder=FakeEmbedder(), vaults="local"))
               for q in QUESTIONS]
        out.append(_snap(ask(Store(s.path), QUESTIONS[0], embedder=FakeEmbedder(),
                             vaults="local", as_of="2026-01-15T00:00:00Z")))
        return out

    cold = answers()                              # sidecar loaded from disk
    (s.path / "views.npz").unlink()
    rebuilt = answers()                           # sidecar refolded from the log
    monkeypatch.setattr(Q, "_views_of", lambda store: None)
    rederived = answers()                         # pre-views derivation, verbatim

    for x, y in zip(cold, rebuilt):
        assert x[:3] == y[:3]
        assert x[3] == pytest.approx(y[3], abs=1e-5)
    for x, y in zip(cold, rederived):
        assert x[:3] == y[:3]
        assert x[3] == pytest.approx(y[3], abs=1e-5)


def test_read_only_store_answers_without_persisting_views(tmp_path):
    s = _curated_store(tmp_path)
    (s.path / "views.npz").unlink()
    s.path.chmod(0o555)
    try:
        ans = ask(Store(s.path), "can we deploy on friday?",
                  embedder=FakeEmbedder(), vaults="local")
        assert ans.hits and not (s.path / "views.npz").exists()
    finally:
        s.path.chmod(0o755)
