"""The relate hop: after the fused ranking picks the hits, the top RELATE_HITS
hits' stored value vectors each probe every OTHER consulted pool — relatedness
across vaults, never verdict material, never an echo from the answer's own
store, ranked hub-penalized so summaries don't crowd out sharp connections."""

import argparse

import numpy as np
import pytest

from mnema.query import (RELATE_FLOOR, RELATE_HITS, RELATE_MAX, RELATE_PER_STORE,
                         RELATE_TOTAL, ask)
from mnema.store import Entry, Store
from mnema.vaults import add_vault

from test_store import ControlledEmbedder, FakeEmbedder, make_store

def _basis(i):
    v = np.zeros(32)
    v[i] = 1.0
    return v


def _toward(c, axis):
    """A unit vector at cosine c to U along `axis`."""
    return c * U + np.sqrt(1 - c ** 2) * axis


U, E1, E2, FAR = _basis(0), _basis(1), _basis(2), _basis(3)
NEAR = _toward(0.97, E1)
NEAR2 = _toward(0.95, E2)

# topics are embedded addresses too — park each on its own axis, away from U
TOPICS = {t: _basis(10 + i) for i, t in enumerate(
    ["doctrine", "miller", "plant", "echo", "lunch", "ruling"])}


def _fold(store, emb, rows):
    store.append([Entry(text=t, topic=k, at=at) for t, k, at in rows])
    store.catch_up(emb, quiet=True)
    return store


def _pair(tmp_path, local_rows, vault_rows, table, vault_name="alice"):
    emb = ControlledEmbedder(table | TOPICS)
    remote = _fold(make_store(tmp_path, "remote"), emb, vault_rows)
    local = _fold(make_store(tmp_path, "local"), emb, local_rows)
    add_vault(local.path, str(remote.path), name=vault_name)
    return local, emb


def _constellation(tmp_path, table, local_rows, vaults):
    """local + named vaults, keyless rows: (text, at) pairs."""
    emb = ControlledEmbedder(table)
    local = make_store(tmp_path, "local")
    local.append([Entry(text=t, at=at) for t, at in local_rows])
    local.catch_up(emb, quiet=True)
    for name, rows in vaults.items():
        remote = make_store(tmp_path, f"remote-{name}")
        remote.append([Entry(text=t, at=at) for t, at in rows])
        remote.catch_up(emb, quiet=True)
        add_vault(local.path, str(remote.path), name=name)
    return local, emb


def test_related_come_only_from_non_origin_pools_and_respect_floor(tmp_path):
    table = {"delete the second copy": U, "what deletes duplicates?": U,
             "authority is fractal; duplicated power erodes": NEAR,
             "the office plant needs watering": FAR}
    local, emb = _pair(
        tmp_path,
        [("delete the second copy", "doctrine", "2026-01-01T00:00:00Z")],
        [("authority is fractal; duplicated power erodes", "miller",
          "2026-01-02T00:00:00Z"),
         ("the office plant needs watering", "plant", "2026-01-03T00:00:00Z")],
        table)

    ans = ask(local, "what deletes duplicates?", embedder=emb, vaults="all", top=1)
    assert ans.hits[0].source == "local"
    assert [r.topic for r in ans.related] == ["miller"]     # far row below floor
    assert all(r.source == "alice" for r in ans.related)
    assert all(r.cos >= RELATE_FLOOR for r in ans.related)


def test_related_dedupe_by_topic_within_a_pool(tmp_path):
    table = {"delete the second copy": U, "what deletes duplicates?": U,
             "authority is fractal": NEAR, "authority is fractal, revised": NEAR2}
    local, emb = _pair(
        tmp_path,
        [("delete the second copy", "doctrine", "2026-01-01T00:00:00Z")],
        [("authority is fractal", "miller", "2026-01-02T00:00:00Z"),
         ("authority is fractal, revised", "miller", "2026-01-03T00:00:00Z")],
        table)

    ans = ask(local, "what deletes duplicates?", embedder=emb, vaults="all", top=1)
    assert [r.topic for r in ans.related] == ["miller"]     # one row per topic
    assert ans.related[0].text == "authority is fractal"    # the closer row


def test_single_store_ask_has_no_related_and_is_otherwise_unchanged(tmp_path):
    s = _fold(make_store(tmp_path), FakeEmbedder(),
              [("the runbook lives in ops/runbooks", "runbook",
                "2026-01-01T00:00:00Z"),
               ("standup is at 4pm", "standup", "2026-01-02T00:00:00Z")])
    a = ask(s, "the runbook lives in ops/runbooks",
            embedder=FakeEmbedder(), vaults="all")
    b = ask(s, "the runbook lives in ops/runbooks",
            embedder=FakeEmbedder(), vaults="local")
    assert a.related == [] and b.related == []
    assert a.related_by_hit == [] and b.related_by_hit == []
    assert (a.verdict, a.hits, a.support) == (b.verdict, b.hits, b.support)


def test_single_store_cli_output_is_byte_identical_across_all_and_local(
        tmp_path, monkeypatch, capsys):
    s = _fold(make_store(tmp_path), FakeEmbedder(),
              [("the runbook lives in ops/runbooks", "runbook",
                "2026-01-01T00:00:00Z"),
               ("standup is at 4pm", "standup", "2026-01-02T00:00:00Z")])
    outs = []
    for local in (False, True):
        outs.append(_run_cli_ask(s, FakeEmbedder(), monkeypatch, capsys,
                                 "the runbook lives in ops/runbooks", top=5,
                                 local_only=local))
    assert outs[0] == outs[1]
    assert "<related " not in outs[0]


def test_vault_origin_top_hit_relates_back_into_local(tmp_path):
    table = {"the ruling text itself": U, "what is the ruling?": U,
             "a local echo of that ruling": NEAR,
             "lunch is at noon": FAR}
    local, emb = _pair(
        tmp_path,
        [("a local echo of that ruling", "echo", "2026-01-01T00:00:00Z"),
         ("lunch is at noon", "lunch", "2026-01-02T00:00:00Z")],
        [("the ruling text itself", "ruling", "2026-01-03T00:00:00Z")],
        table)

    ans = ask(local, "what is the ruling?", embedder=emb, vaults="all", top=1)
    assert ans.hits[0].source == "alice"
    assert [(r.source, r.topic) for r in ans.related] == [("local", "echo")]


def test_related_survive_the_legacy_as_of_path(tmp_path):
    table = {"delete the second copy": U, "what deletes duplicates?": U,
             "authority is fractal; duplicated power erodes": NEAR}
    local, emb = _pair(
        tmp_path,
        [("delete the second copy", "doctrine", "2026-01-01T00:00:00Z")],
        [("authority is fractal; duplicated power erodes", "miller",
          "2026-01-02T00:00:00Z")],
        table)

    ans = ask(local, "what deletes duplicates?", embedder=emb, vaults="all",
              as_of="2027-01-01T00:00:00Z", top=1)
    assert [(r.source, r.topic) for r in ans.related] == [("alice", "miller")]


def test_shown_hits_are_never_related_to_each_other(tmp_path):
    table = {"delete the second copy": U, "what deletes duplicates?": U,
             "authority is fractal; duplicated power erodes": NEAR}
    local, emb = _pair(
        tmp_path,
        [("delete the second copy", "doctrine", "2026-01-01T00:00:00Z")],
        [("authority is fractal; duplicated power erodes", "miller",
          "2026-01-02T00:00:00Z")],
        table)
    ans = ask(local, "what deletes duplicates?", embedder=emb, vaults="all", top=5)
    assert {h.source for h in ans.hits} == {"local", "alice"}   # both shown
    assert all(rel == [] for rel in ans.related_by_hit)


# ---- selection caps: per store, per hit, per ask -------------------------

def _flood(prefix, n, axis_from):
    """n distinct rows around U (cos 0.80..0.90), each on its own axis."""
    return {f"{prefix} {i}": _toward(0.80 + 0.01 * i, _basis(axis_from + i))
            for i in range(n)}


def test_per_store_and_total_caps_leave_a_third_vault_its_voice(tmp_path):
    table = {"the answer": U, "the question?": U}
    a, b = _flood("alpha", 6, 4), _flood("beta", 6, 12)
    c = {"gamma only": _toward(0.85, _basis(20))}
    table |= a | b | c
    rows = lambda d: [(t, "2026-01-02T00:00:00Z") for t in d]
    local, emb = _constellation(
        tmp_path, table, [("the answer", "2026-01-01T00:00:00Z")],
        {"alpha": rows(a), "beta": rows(b), "gamma": rows(c)})

    ans = ask(local, "the question?", embedder=emb, vaults="all", top=1)
    by_source = {}
    for r in ans.related:
        by_source[r.source] = by_source.get(r.source, 0) + 1
    assert by_source == {"alpha": 2, "beta": 2, "gamma": 1}
    assert len(ans.related) <= RELATE_TOTAL
    assert max(by_source.values()) <= RELATE_PER_STORE
    assert all(r.cos >= RELATE_FLOOR for r in ans.related)


def test_related_max_bounds_the_ask_across_three_hits(tmp_path):
    hits = {f"hit {i}": _toward(0.99, _basis(1 + i)) for i in range(3)}
    a, b, c = _flood("alpha", 6, 4), _flood("beta", 6, 12), _flood("gamma", 6, 20)
    table = {"the question?": U} | hits | a | b | c
    rows = lambda d: [(t, "2026-01-02T00:00:00Z") for t in d]
    local, emb = _constellation(
        tmp_path, table, [(t, "2026-01-01T00:00:00Z") for t in hits],
        {"alpha": rows(a), "beta": rows(b), "gamma": rows(c)})

    ans = ask(local, "the question?", embedder=emb, vaults="all", top=5)
    assert [h.text for h in ans.hits[:3]] == sorted(hits)[:3] or \
        {h.text for h in ans.hits[:3]} == set(hits)          # local rows lead
    assert len(ans.related_by_hit) == RELATE_HITS
    total = sum(len(rel) for rel in ans.related_by_hit)
    assert total == RELATE_MAX                               # 6 + 6 + 0
    assert [len(rel) for rel in ans.related_by_hit] == [6, 6, 0]
    hashes = [r.h for rel in ans.related_by_hit for r in rel]
    assert len(hashes) == len(set(hashes))                   # cross-hit dedupe
    shown = {h.h for h in ans.hits}
    assert not shown & set(hashes)


def test_cross_hit_dedupe_shows_a_shared_relation_once_under_the_earlier_hit(
        tmp_path):
    table = {"the question?": U,
             "hit one": _toward(0.95, E1), "hit two": _toward(0.90, E2),
             "shared relation": _toward(0.85, FAR),         # ≥ floor to both hits
             "hit-two-only relation": 0.6 * U + 0.8 * E2}   # 0.89 to hit two, 0.57 to hit one
    local, emb = _constellation(
        tmp_path, table,
        [("hit one", "2026-01-01T00:00:00Z"), ("hit two", "2026-01-01T00:00:01Z")],
        {"alice": [("shared relation", "2026-01-02T00:00:00Z"),
                   ("hit-two-only relation", "2026-01-02T00:00:01Z")]})
    ans = ask(local, "the question?", embedder=emb, vaults="all", top=2)
    assert [h.text for h in ans.hits] == ["hit one", "hit two"]
    assert [r.text for r in ans.related_by_hit[0]] == ["shared relation"]
    assert [r.text for r in ans.related_by_hit[1]] == ["hit-two-only relation"]


# ---- the ranking law: hub-penalized ------------------------------------

def test_sharp_connection_outranks_a_hub_of_equal_raw_cosine(tmp_path):
    """y_hub sits among many rows of its own store (a summary near all of
    its document); y_sharp is as close to x but far from its store's other
    rows. Raw cosine ties (the hub is even nominally closer); the ranking
    must still put the sharp connection first."""
    y_hub = _toward(0.86, E1)
    y_sharp = _toward(0.84, E2)
    cluster = {f"hub neighbour {i}": 0.6 * U + 0.8 * E1
               + 0.05 * _basis(4 + i) for i in range(8)}    # cos to U ≈ 0.6 (<floor)
    # a broad background (the corpus cone) so the union mean direction is not
    # the cluster itself — centering removes what everything shares, no more
    background = {f"background {i}": _basis(6 + i % 26) for i in range(208)}
    table = {"x": U, "x?": U, "y hub": y_hub, "y sharp": y_sharp} | cluster | background
    local, emb = _constellation(
        tmp_path, table,
        [("x", "2026-01-01T00:00:00Z")]
        + [(t, "2025-01-01T00:00:00Z") for t in background],
        {"alice": [("y hub", "2026-01-02T00:00:00Z"),
                   ("y sharp", "2026-01-02T00:00:01Z")]
                  + [(t, "2026-01-03T00:00:00Z") for t in cluster]})
    ans = ask(local, "x?", embedder=emb, vaults="all", top=1)
    texts = [r.text for r in ans.related]
    assert texts == ["y sharp", "y hub"]                    # sharp first
    hub, sharp = (next(r for r in ans.related if r.text == t)
                  for t in ("y hub", "y sharp"))
    assert hub.cos >= sharp.cos - 0.01                      # raw would not rank it so
    assert sharp.score > hub.score
    assert all(r.cos >= RELATE_FLOOR for r in ans.related)  # neighbours below floor


# ---- --except -----------------------------------------------------------

def _two_vaults(tmp_path):
    table = {"the answer": U, "the question?": U,
             "alice knows": _toward(0.9, E1), "bob knows": _toward(0.9, E2)}
    return _constellation(
        tmp_path, table, [("the answer", "2026-01-01T00:00:00Z")],
        {"alice": [("alice knows", "2026-01-02T00:00:00Z")],
         "bob": [("bob knows", "2026-01-02T00:00:00Z")]})


def test_except_drops_a_vault_from_hits_and_related(tmp_path):
    local, emb = _two_vaults(tmp_path)
    full = ask(local, "the question?", embedder=emb, vaults="all", top=5)
    assert {h.source for h in full.hits} == {"local", "alice", "bob"}
    ans = ask(local, "the question?", embedder=emb, vaults="all", top=1,
              exclude=("bob",))
    assert [h.source for h in ans.hits] == ["local"]
    assert [r.source for r in ans.related] == ["alice"]
    with pytest.raises(SystemExit) as ei:
        ask(local, "the question?", embedder=emb, vaults="all", exclude=("zed",))
    assert "zed" in str(ei.value) and "vault list" in str(ei.value)


def test_cli_except_refuses_to_combine_with_from(tmp_path, monkeypatch):
    local, emb = _two_vaults(tmp_path)
    from mnema import cli
    monkeypatch.setattr("mnema.query.Embedder", lambda model: emb)
    with pytest.raises(SystemExit) as ei:
        cli.cmd_ask(argparse.Namespace(
            store=str(local.path), question="the question?", top=5,
            as_of=None, current=False, slot=None, scores=False, local=False,
            from_vault="alice", except_vaults=["bob"]))
    assert "pick one" in str(ei.value)


# ---- rendering ---------------------------------------------------------

def _run_cli_ask(local, emb, monkeypatch, capsys, question, top=1,
                 local_only=False, scores=False):
    from mnema import cli
    monkeypatch.setattr("mnema.query.Embedder", lambda model: emb)
    cli.cmd_ask(argparse.Namespace(
        store=str(local.path), question=question, top=top,
        as_of=None, current=False, slot=None, scores=scores,
        local=local_only, from_vault=None, except_vaults=[]))
    return capsys.readouterr().out


def test_cli_prints_related_as_one_self_closing_line_inside_the_first_hit(
        tmp_path, monkeypatch, capsys):
    table = {"delete the second copy": U, "what deletes duplicates?": U,
             "keep exactly one copy": _toward(0.985, FAR),
             "authority is fractal; duplicated power erodes": NEAR}
    local, emb = _pair(
        tmp_path,
        [("delete the second copy", "doctrine", "2026-01-01T00:00:00Z"),
         ("keep exactly one copy", "echo", "2026-01-01T00:00:01Z")],
        [("authority is fractal; duplicated power erodes", "miller",
          "2026-01-02T00:00:00Z")],
        table)

    out = _run_cli_ask(local, emb, monkeypatch, capsys,
                       "what deletes duplicates?", top=2)
    relate = next(l for l in out.splitlines() if "<related " in l)
    vault_h = Entry(text="authority is fractal; duplicated power erodes",
                    topic="miller", at="2026-01-02T00:00:00Z").h
    second_h = Entry(text="keep exactly one copy", topic="echo",
                     at="2026-01-01T00:00:01Z").h
    assert relate.startswith("  <related ") and relate.endswith("/>")
    assert f'h="{vault_h[:8]}"' in relate
    assert 'vault="alice"' in relate and 'cos="0.9' in relate
    assert 'gloss="miller"' in relate                    # a topic IS the gloss
    assert "</related>" not in out                       # no body, no closing tag
    assert "authority is fractal" not in out.split("</hit>")[0]  # no snippet
    first_hit_close = out.index("</hit>")
    assert out.index("<related ") < first_hit_close      # inside hit 1 only
    assert out.index(f'<hit h="{second_h[:8]}"') > first_hit_close  # hit 2 after
    assert out.count("<related ") == 1                   # related to hit 2 too,
    assert out.count("<hit ") == 2                       # but shown once, under hit 1


def test_cli_gloss_for_keyless_relations_is_the_opening_words(
        tmp_path, monkeypatch, capsys):
    long = " ".join(f"w{i}" for i in range(30))
    table = {"the answer": U, "the question?": U, long: NEAR}
    local, emb = _constellation(
        tmp_path, table, [("the answer", "2026-01-01T00:00:00Z")],
        {"alice": [(long, "2026-01-02T00:00:00Z")]})
    out = _run_cli_ask(local, emb, monkeypatch, capsys, "the question?")
    relate = next(l for l in out.splitlines() if "<related " in l)
    assert f'gloss="{" ".join(f"w{i}" for i in range(15))}…"' in relate
    assert "w29" not in out
    scored = _run_cli_ask(local, emb, monkeypatch, capsys, "the question?",
                          scores=True)
    assert ' score="' in next(l for l in scored.splitlines() if "<related " in l)


def test_cli_navigate_line_raw_attributes_and_three_related(
        tmp_path, monkeypatch, capsys):
    weird = 'rule <"x"> & y'
    third = _toward(0.90, _basis(4))
    table = {"delete the second copy": U, "what deletes duplicates?": U,
             "authority is fractal; duplicated power erodes": NEAR,
             "power fragments under duplication": NEAR2,
             "duplicates decay the source": third,
             weird: _basis(20), "t-one": _basis(21), "t-two": _basis(22),
             "t-three": _basis(23)}
    local, emb = _pair(
        tmp_path,
        [("delete the second copy", weird, "2026-01-01T00:00:00Z")],
        [("authority is fractal; duplicated power erodes", "t-one",
          "2026-01-02T00:00:00Z"),
         ("power fragments under duplication", "t-two", "2026-01-03T00:00:00Z")],
        table)
    bob = _fold(make_store(tmp_path, "remote-bob"), emb,
                [("duplicates decay the source", "t-three", "2026-01-04T00:00:00Z")])
    add_vault(local.path, str(bob.path), name="bob")

    out = _run_cli_ask(local, emb, monkeypatch, capsys,
                       "what deletes duplicates?")
    lines = out.splitlines()
    assert lines[1].startswith("navigate: ")             # present and second
    assert "mnema show <hash>" in lines[1] and "--except <vault>" in lines[1]
    assert f'topic="{weird}"' in out                     # raw, never escaped
    assert not any(esc in out for esc in ("&amp;", "&lt;", "&gt;", "&quot;"))
    assert out.count("<related ") == 3                   # three clear the floor
    relates = [l for l in lines if "<related " in l]
    assert all(l.endswith("/>") for l in relates)
    assert sum('vault="alice"' in l for l in relates) == 2
    assert sum('vault="bob"' in l for l in relates) == 1
    assert {l.split('gloss="')[1][:-3] for l in relates} == {"t-one", "t-two", "t-three"}


def test_cli_nests_related_under_each_of_the_top_three_hits_only(
        tmp_path, monkeypatch, capsys):
    hits = {f"hit {i}": _toward(0.99, _basis(1 + i)) for i in range(4)}
    a = _flood("alpha", 4, 6)
    table = {"the question?": U} | hits | a
    local, emb = _constellation(
        tmp_path, table, [(t, "2026-01-01T00:00:00Z") for t in hits],
        {"alpha": [(t, "2026-01-02T00:00:00Z") for t in a]})
    out = _run_cli_ask(local, emb, monkeypatch, capsys, "the question?", top=4)
    blocks = out.split("\n<hit ")[1:]
    assert len(blocks) == 4
    per_block = [b.count("<related ") for b in blocks]
    assert per_block[:2] == [2, 2] and per_block[3] == 0     # 4th hit carries none
    assert sum(per_block) == 4                               # dedupe drained the pool
