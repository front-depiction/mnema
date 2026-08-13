"""`mnema show`: any printed hash prefix dereferences to the full memory —
the local store first, then every mounted vault's mirror. A pure log parse:
no Embedder is ever constructed."""

import argparse
from collections import Counter

import pytest

from mnema import cli
from mnema.store import Entry
from mnema.vaults import add_vault

from test_store import make_store


LONG = ("the settlement gate folds the admitted prefix before the turn "
        "closes, so a batch that arrives mid-turn lands behind the fold and "
        "is admitted on the next turn; grouping a causal sequence into "
        "batches must not change the resulting state, which is exactly the "
        "batch-invariance law the suite pins")


def _ns(store, *hashes):
    return argparse.Namespace(store=str(store.path), hashes=list(hashes))


def _forbid_embedder(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("show must be model-free: Embedder constructed")
    monkeypatch.setattr("mnema.embed.Embedder.__init__", boom)


def test_show_resolves_a_local_prefix_to_the_full_untruncated_memory(
        tmp_path, monkeypatch, capsys):
    _forbid_embedder(monkeypatch)
    s = make_store(tmp_path)
    e = Entry(text=LONG, at="2026-01-01T00:00:00Z")
    s.append([e])

    cli.cmd_show(_ns(s, e.h[:8]))
    out = capsys.readouterr().out
    assert e.h in out                                    # full 16-char hash
    assert "(local)" in out and "[note]" in out
    assert "".join(LONG.split()) in "".join(out.split())  # wrapped, never cut


def test_show_annotates_supersession_displacement_and_questions(
        tmp_path, monkeypatch, capsys):
    _forbid_embedder(monkeypatch)
    s = make_store(tmp_path)
    a = Entry(text="policy is A", topic="policy", at="2026-01-01T00:00:00Z")
    b = Entry(text="policy is B now", topic="policy", at="2026-02-01T00:00:00Z",
              questions=["what is the policy?"])
    c = Entry(text="policy A is retired", at="2026-03-01T00:00:00Z",
              displaces=[[a.h, 0.83]])
    s.append([a, b, c])

    cli.cmd_show(_ns(s, a.h[:8]))
    out = capsys.readouterr().out
    assert f"superseded by {b.h}" in out
    assert "displaced 0.83 by a later write" in out

    cli.cmd_show(_ns(s, c.h[:8]))
    out = capsys.readouterr().out
    assert f"displaces {a.h} [0.83]" in out and "policy is A" in out

    cli.cmd_show(_ns(s, b.h[:8]))
    out = capsys.readouterr().out
    assert out.count('question: "what is the policy?"') == 1

    alias = next(x for x in s.entries() if x.kind == "alias")
    cli.cmd_show(_ns(s, alias.h[:8]))                    # aliases are findable
    out = capsys.readouterr().out
    assert "[alias]" in out and f"alias of {b.h}" in out


def test_show_resolves_a_vault_only_hash_and_reports_its_origin(
        tmp_path, monkeypatch, capsys):
    _forbid_embedder(monkeypatch)
    remote = make_store(tmp_path, "remote")
    ruling = Entry(text=LONG, topic="ruling", at="2026-01-02T00:00:00Z")
    remote.append([ruling])
    local = make_store(tmp_path, "local")
    local.append([Entry(text="a local note", at="2026-01-01T00:00:00Z")])
    add_vault(local.path, str(remote.path), name="alice")

    cli.cmd_show(_ns(local, ruling.h[:8]))
    out = capsys.readouterr().out
    assert ruling.h in out and "(alice)" in out and "ruling" in out
    assert "".join(LONG.split()) in "".join(out.split())


def test_show_ambiguous_prefix_lists_candidates_and_exits_nonzero(tmp_path):
    s = make_store(tmp_path)
    entries = [Entry(text=f"note {i}", at=f"2026-01-01T00:00:{i:02d}Z")
               for i in range(20)]
    s.append(entries)
    char = next(ch for ch, n in Counter(e.h[0] for e in entries).items()
                if n >= 2)
    dupes = [e for e in entries if e.h[0] == char]

    with pytest.raises(SystemExit) as ei:
        cli.cmd_show(_ns(s, char))
    msg = str(ei.value)
    assert ei.value.code                                 # nonzero exit
    assert char in msg and "ambiguous" in msg
    for e in dupes:
        assert f"{e.h}  [note] —  (local)" in msg        # requalifiable lines


def test_show_unknown_prefix_exits_nonzero_and_names_it(tmp_path):
    s = make_store(tmp_path)
    s.append([Entry(text="one note", at="2026-01-01T00:00:00Z")])
    with pytest.raises(SystemExit) as ei:
        cli.cmd_show(_ns(s, "zzzz9999"))
    assert ei.value.code and "zzzz9999" in str(ei.value)
