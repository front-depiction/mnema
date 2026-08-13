"""The relate hop: after the fused ranking picks the top hit, its stored value
vector probes every OTHER consulted pool — relatedness across vaults, never
verdict material, and never an echo from the answer's own store."""

import argparse

import numpy as np

from mnema.query import RELATE_FLOOR, ask
from mnema.store import Entry, Store
from mnema.vaults import add_vault

from test_store import ControlledEmbedder, FakeEmbedder, make_store

def _basis(i):
    v = np.zeros(32)
    v[i] = 1.0
    return v


U, E1, E2, FAR = _basis(0), _basis(1), _basis(2), _basis(3)
NEAR = 0.97 * U + np.sqrt(1 - 0.97**2) * E1
NEAR2 = 0.95 * U + np.sqrt(1 - 0.95**2) * E2

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

    ans = ask(local, "what deletes duplicates?", embedder=emb, vaults="all")
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

    ans = ask(local, "what deletes duplicates?", embedder=emb, vaults="all")
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
    assert (a.verdict, a.hits) == (b.verdict, b.hits)


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

    ans = ask(local, "what is the ruling?", embedder=emb, vaults="all")
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
              as_of="2027-01-01T00:00:00Z")
    assert [(r.source, r.topic) for r in ans.related] == [("alice", "miller")]


def _run_cli_ask(local, emb, monkeypatch, capsys, question):
    from mnema import cli
    monkeypatch.setattr("mnema.query.Embedder", lambda model: emb)
    cli.cmd_ask(argparse.Namespace(
        store=str(local.path), question=question, top=5,
        as_of=None, current=False, slot=None, scores=False, local=False,
        from_vault=None))
    return capsys.readouterr().out


def test_cli_prints_related_blocks_inside_the_first_hit(tmp_path, monkeypatch, capsys):
    table = {"delete the second copy": U, "what deletes duplicates?": U,
             "authority is fractal; duplicated power erodes": NEAR}
    local, emb = _pair(
        tmp_path,
        [("delete the second copy", "doctrine", "2026-01-01T00:00:00Z")],
        [("authority is fractal; duplicated power erodes", "miller",
          "2026-01-02T00:00:00Z")],
        table)

    out = _run_cli_ask(local, emb, monkeypatch, capsys,
                       "what deletes duplicates?")
    relate = next(l for l in out.splitlines() if "<related " in l)
    assert 'vault="alice"' in relate and 'cos="0.9' in relate
    assert 'topic="miller"' in relate
    vault_h = Entry(text="authority is fractal; duplicated power erodes",
                    topic="miller", at="2026-01-02T00:00:00Z").h
    assert f'h="{vault_h[:8]}"' in relate                # hash present
    assert "authority is fractal; duplicated power erodes" in out  # full snippet
    first_hit_close = out.index("</hit>")
    assert out.index("<related ") < first_hit_close      # inside hit 1 only
    assert out.index("</related>") < first_hit_close
    assert out.index(f'<hit h="{vault_h[:8]}"') > first_hit_close  # hit 2 after


def test_cli_navigate_line_raw_attributes_and_three_related(
        tmp_path, monkeypatch, capsys):
    weird = 'rule <"x"> & y'
    third = 0.90 * U + np.sqrt(1 - 0.90**2) * _basis(4)
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
         ("power fragments under duplication", "t-two", "2026-01-03T00:00:00Z"),
         ("duplicates decay the source", "t-three", "2026-01-04T00:00:00Z")],
        table)

    out = _run_cli_ask(local, emb, monkeypatch, capsys,
                       "what deletes duplicates?")
    lines = out.splitlines()
    assert lines[1].startswith("navigate: ")             # present and second
    assert "mnema show <hash>" in lines[1]
    assert f'topic="{weird}"' in out                     # raw, never escaped
    assert not any(esc in out for esc in ("&amp;", "&lt;", "&gt;", "&quot;"))
    assert out.count("<related ") == 3                   # three clear the floor
    assert out.count("</related>") == 3
    relates = [l for l in lines if "<related " in l]
    assert all('vault="alice"' in l for l in relates)
