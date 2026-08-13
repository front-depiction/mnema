"""Paper translator behavior: horizon-safe packing, disjoint slots, navigation
dropping, determinism, and the ingest --at override — model-free, no anydoc."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from mnema.store import Store, past_horizon
from mnema.translate import paper


def para(word, n):
    return " ".join(f"{word}{i}" for i in range(n))


def write_md(tmp_path, text, name="paper.md"):
    f = tmp_path / name
    f.write_text(text)
    return f


def test_long_section_packs_into_horizon_sized_chunks(tmp_path):
    body = "\n\n".join(para(f"p{i}", 60) for i in range(12))     # 720 words
    f = write_md(tmp_path, f"# Long Section\n\n{body}\n")
    entries = paper(f)
    assert len(entries) > 1
    assert all(not past_horizon(e.text) for e in entries)
    assert [e.topic for e in entries] == [
        f"{f}#long-section", f"{f}#long-section/2", f"{f}#long-section/3"]
    assert all(e.text.startswith("Long Section\n") for e in entries)
    assert all(e.kind == "doc" for e in entries)


def test_short_section_stays_single_chunk_without_suffix(tmp_path):
    f = write_md(tmp_path, f"# Short\n\n{para('s', 40)}\n")
    entries = paper(f)
    assert [e.topic for e in entries] == [f"{f}#short"]


def test_oversized_single_paragraph_stays_whole(tmp_path):
    f = write_md(tmp_path, f"# Wall\n\n{para('w', 400)}\n")
    entries = paper(f)
    assert [e.topic for e in entries] == [f"{f}#wall"]
    assert past_horizon(entries[0].text)                          # warned downstream


def test_navigation_sections_are_dropped(tmp_path):
    nav = "\n".join(f"{i} Chapter Title Number {i} .......... {i * 7}" for i in range(1, 9))
    f = write_md(tmp_path, f"# Contents\n\n{nav}\n\n"
                           f"## Table of Contents\n\n{nav}\n\n"
                           f"# Abstract\n\n{para('a', 60)}\n")
    entries = paper(f)
    assert [e.topic for e in entries] == [f"{f}#abstract"]


def test_topics_are_disjoint_across_all_chunks(tmp_path):
    one = "\n\n".join(para(f"one{i}", 60) for i in range(8))
    two = "\n\n".join(para(f"two{i}", 60) for i in range(8))
    f = write_md(tmp_path, f"{para('pre', 60)}\n\n# One\n\n{one}\n\n# Two\n\n{two}\n")
    entries = paper(f)
    topics = [e.topic for e in entries]
    assert len(set(topics)) == len(topics) >= 5


def test_translation_is_deterministic(tmp_path):
    body = "\n\n".join(para(f"a{i}", 60) for i in range(8))
    f = write_md(tmp_path, f"# A\n\n{body}\n")
    a, b = paper(f), paper(f)
    assert [(e.topic, e.text, e.at, e.h) for e in a] == \
           [(e.topic, e.text, e.at, e.h) for e in b]


def test_missing_anydoc_exits_with_remedy(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "anydoc", None)              # import raises
    f = tmp_path / "x.pdf"
    f.write_bytes(b"%PDF-1.4")
    with pytest.raises(SystemExit, match="firecrawl-anydoc"):
        paper(f)


class FakeEmbedder:
    dim = 32

    def __init__(self, model_name="fake", direct=False):
        pass

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


def test_ingest_at_override_stamps_every_entry(tmp_path, monkeypatch):
    import mnema.store as store_mod
    from mnema.cli import cmd_ingest
    monkeypatch.setattr(store_mod, "Embedder", FakeEmbedder)
    f = write_md(tmp_path, f"# Alpha\n\n{para('alpha', 60)}\n\n"
                           f"# Beta\n\n{para('beta', 60)}\n")
    store = Store.init(tmp_path / "s", model="fake", dim=64, beta=1.0,
                       sigma=None, seed=7, d_embed=32)
    cmd_ingest(SimpleNamespace(store=str(store.path), format="paper", infer=False,
                               at="2020-05-01T00:00:00Z", paths=[str(f)]))
    entries = Store(store.path).entries()
    assert len(entries) == 2
    assert all(e.at == "2020-05-01T00:00:00Z" for e in entries)
