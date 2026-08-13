"""Vault behavior: read fan-out across stores, targeted asks, write isolation."""

import numpy as np

from mnema.query import ask
from mnema.store import Entry, Store
from mnema.vaults import add_vault, load_vaults, sync_vault

from test_store import FakeEmbedder, make_store


def _seed(tmp_path, name, text, topic, at):
    s = make_store(tmp_path, name)
    s.append([Entry(text=text, topic=topic, at=at)])
    s.catch_up(FakeEmbedder(), quiet=True)
    return s


def test_vault_fanout_and_source_tagging(tmp_path):
    remote = _seed(tmp_path, "remote", "the runbook lives in ops/runbooks",
                   "runbook", "2026-01-01T00:00:00Z")
    local = _seed(tmp_path, "local", "standup is at 4pm", "standup",
                  "2026-01-02T00:00:00Z")
    add_vault(local.path, str(remote.path), name="alice")       # DirSource adapter

    ans = ask(local, "the runbook lives in ops/runbooks",
              embedder=FakeEmbedder(), vaults="all")
    assert any(h.source == "alice" and h.topic == "runbook" for h in ans.hits)
    assert any(h.source == "local" for h in ans.hits)
    assert ans.warnings == []


def test_vault_targeted_ask_via_file_url(tmp_path):
    remote = _seed(tmp_path, "remote2", "incident channel is #sev",
                   "incidents", "2026-01-01T00:00:00Z")
    local = _seed(tmp_path, "local2", "standup is at 4pm", "standup",
                  "2026-01-02T00:00:00Z")
    add_vault(local.path, remote.path.as_uri(), name="bob")     # UrlSource adapter

    ans = ask(local, "incident channel is #sev",
              embedder=FakeEmbedder(), vaults="bob")
    assert all(h.source == "bob" for h in ans.hits)             # local excluded
    assert any(h.topic == "incidents" for h in ans.hits)


def test_writes_never_touch_vaults(tmp_path):
    remote = _seed(tmp_path, "remote3", "a remote fact", "fact",
                   "2026-01-01T00:00:00Z")
    local = _seed(tmp_path, "local3", "a local fact", "localfact",
                  "2026-01-02T00:00:00Z")
    add_vault(local.path, str(remote.path), name="peer")
    sync_vault(local.path, load_vaults(local.path)[0])
    before = (remote.path / "log.jsonl").read_bytes()

    local.append([Entry(text="another local fact", topic="more",
                        at="2026-01-03T00:00:00Z")])
    local.catch_up(FakeEmbedder(), quiet=True)

    assert (remote.path / "log.jsonl").read_bytes() == before   # untouched
    assert len(Store(local.path / "vaults" / "peer").entries()) == 1


def test_vault_config_mismatch_is_skipped_with_warning(tmp_path):
    remote = Store.init(tmp_path / "other", model="fake", dim=128, beta=1.0,
                        sigma=None, seed=7, d_embed=32)
    remote.append([Entry(text="x", at="2026-01-01T00:00:00Z")])
    remote.catch_up(FakeEmbedder(), quiet=True)
    local = _seed(tmp_path, "local4", "y", None, "2026-01-02T00:00:00Z")
    add_vault(local.path, str(remote.path), name="odd")

    ans = ask(local, "y", embedder=FakeEmbedder(), vaults="all")
    assert any("odd" in w and "dim" in w for w in ans.warnings)
    assert all(h.source == "local" for h in ans.hits)
