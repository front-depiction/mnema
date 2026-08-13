"""Read-side views: the query-time component of the fused product fold.

Everything ask() needs that is derivable from the log — the inverted lexical
index, latest-by-topic, retraction/revocation sets, alias parent links, and
the currency cluster inputs — advances here as one fold over the same log the
matrix state folds, and persists as a sidecar next to state.npz. The license
for the cache is the law the tests assert: advancing in segments equals
refolding from scratch, for any segmentation, so cached == recomputed.

Like the matrix fold, a keep/retract event in the unfolded tail forces a
clean refold from zero — revocation is exact by re-derivation, never by
patching an index in place.

The inverted index is columnar and append-only: parallel (term, row, tf)
arrays in row order plus a term table, so advancing is pure array append and
loading is a binary read. Query-time lookup scans post_term once per query
term — O(postings) vectorized, far below the corpus scan it replaces."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .lexical import tokenize

_EMPTY_I32 = np.empty(0, np.int32)


@dataclass
class Views:
    n: int                                # rows folded
    post_term: np.ndarray                 # int32 [P] term id per posting
    post_row: np.ndarray                  # int32 [P]
    post_tf: np.ndarray                   # int32 [P]
    terms: list[str]                      # term id -> term
    term_id: dict[str, int]               # term -> id (derived from terms)
    doc_len: np.ndarray                   # int32 [n]; 0 for rows outside the corpus
    mask: np.ndarray                      # bool [n]: live at readout
    latest_by_topic: dict[str, int]       # declared topic -> latest live row
    topic_groups: dict[str, list[int]]    # declared topic -> live rows, in order
    disp_pairs: list                      # [displacer_row, target_row] live edges
    alias_parent: list                    # [alias_row, parent_row], row order
    displaced: dict[str, float]           # target hash -> strongest live weight
    retracted: set[str] = field(default_factory=set)
    revoked: set[str] = field(default_factory=set)   # keep targets

    @classmethod
    def empty(cls) -> "Views":
        return cls(0, _EMPTY_I32, _EMPTY_I32, _EMPTY_I32, [], {},
                   _EMPTY_I32, np.empty(0, bool), {}, {}, [], [], {})


def advance(views: Views, entries: list, start: int) -> Views:
    """Fold entries[start:] into the views; start must be views.n. A keep or
    retract in the tail acts on past occurrences too, so the fold restarts
    from zero — which is also what makes any segmentation lawful."""
    end = len(entries)
    if any(e.kind in ("keep", "retract") for e in entries[start:end]):
        views, start = Views.empty(), 0
    if start >= end:
        return views

    retracted = set(views.retracted)
    revoked = set(views.revoked)
    for e in entries[start:end]:
        if e.target and e.kind == "retract":
            retracted.add(e.target)
        elif e.target and e.kind == "keep":
            revoked.add(e.target)

    terms = list(views.terms)
    term_id = dict(views.term_id)
    pt: list[int] = []
    pr: list[int] = []
    pf: list[int] = []
    doc_len: list[int] = []
    mask: list[bool] = []
    latest = dict(views.latest_by_topic)
    groups = {t: list(g) for t, g in views.topic_groups.items()}
    disp_pairs = list(views.disp_pairs)
    alias_parent = list(views.alias_parent)
    displaced = dict(views.displaced)
    # Alias/displacement targets can sit anywhere in the prefix, but resolving
    # them needs a hash->row map whose build recomputes every prior entry's
    # hash — only pay that when the tail actually references rows.
    need_rows = any(e.kind == "alias" or e.displaces for e in entries[start:end])
    row_of = {e.h: i for i, e in enumerate(entries[:start])} if need_rows else {}

    for i in range(start, end):
        e = entries[i]
        live = e.kind not in ("keep", "retract", "alias") and e.h not in retracted
        toks = tokenize((e.topic or "") + " " + e.text) if live else []
        doc_len.append(len(toks))
        mask.append(live)
        if toks:
            for w, c in Counter(toks).items():
                tid = term_id.get(w)
                if tid is None:
                    tid = term_id[w] = len(terms)
                    terms.append(w)
                pt.append(tid)
                pr.append(i)
                pf.append(c)
        if live and e.topic:
            groups.setdefault(e.topic, []).append(i)
            latest[e.topic] = i
        if e.kind == "alias" and e.target in row_of:
            alias_parent.append([i, row_of[e.target]])
        if e.h not in retracted:
            for target_h, w in e.displaces:
                if target_h in revoked:
                    continue
                displaced[target_h] = max(displaced.get(target_h, 0.0), float(w))
                if target_h in row_of and target_h not in retracted:
                    disp_pairs.append([i, row_of[target_h]])
        if need_rows:
            row_of[e.h] = i

    cat = lambda old, new: np.concatenate([old, np.asarray(new, np.int32)])
    return Views(end, cat(views.post_term, pt), cat(views.post_row, pr),
                 cat(views.post_tf, pf), terms, term_id,
                 cat(views.doc_len, doc_len),
                 np.concatenate([views.mask, np.asarray(mask, bool)]),
                 latest, groups, disp_pairs, alias_parent, displaced,
                 retracted, revoked)


def of(entries: list) -> Views:
    return advance(Views.empty(), entries, 0)


def postings_of(views: Views, term: str) -> tuple[np.ndarray, np.ndarray]:
    """(rows, tf) of every doc containing the term; df = len(rows)."""
    tid = views.term_id.get(term)
    if tid is None:
        return _EMPTY_I32, _EMPTY_I32
    sel = views.post_term == tid
    return views.post_row[sel], views.post_tf[sel]


def save(views: Views, path: Path) -> None:
    meta = {"latest_by_topic": views.latest_by_topic,
            "topic_groups": views.topic_groups,
            "disp_pairs": views.disp_pairs,
            "alias_parent": views.alias_parent,
            "displaced": views.displaced,
            "retracted": sorted(views.retracted),
            "revoked": sorted(views.revoked)}
    np.savez(path, n=views.n, post_term=views.post_term,
             post_row=views.post_row, post_tf=views.post_tf,
             doc_len=views.doc_len, mask=views.mask,
             vocab=np.frombuffer("\n".join(views.terms).encode(), np.uint8),
             meta=np.frombuffer(json.dumps(meta).encode(), np.uint8))


def load(path: Path) -> Views:
    with np.load(path) as z:
        vocab = bytes(z["vocab"]).decode()
        terms = vocab.split("\n") if vocab else []
        meta = json.loads(bytes(z["meta"]).decode())
        return Views(int(z["n"]), z["post_term"], z["post_row"], z["post_tf"],
                     terms, dict(zip(terms, range(len(terms)))),
                     z["doc_len"], z["mask"],
                     meta["latest_by_topic"], meta["topic_groups"],
                     meta["disp_pairs"], meta["alias_parent"],
                     meta["displaced"],
                     set(meta["retracted"]), set(meta["revoked"]))
