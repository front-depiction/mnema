"""The lexical view: exact-term evidence with jurisdiction gating.

Dense embeddings and lexical matching have near-disjoint error modes — dense
blurs rare/coined terms into paraphrase-soup; lexical is blind to paraphrase
but surgical on jargon. Fusing them multiplies independent evidence, and the
IDF gate confines the lexical view to its jurisdiction: it only votes when
the question contains a term that is genuinely rare in the visible corpus.

All functions are pure and pool-global: BM25 statistics and the gate must be
computed over the UNION of every store consulted by a query (rarity and rank
are only meaningful against everything you can see at once)."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

GATE_FLOOR = 3.0   # max-IDF below this: lexical view abstains entirely
GATE_SPAN = 3.0    # IDF above floor+span: lexical view at full weight
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens; hyphenated compounds emit both the
    compound and its parts, so "level-triggered" matches "level triggered"."""
    out: list[str] = []
    for w in re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower()):
        out.append(w)
        if "-" in w:
            out.extend(w.split("-"))
    return out


def bm25(doc_tokens: list[list[str]], q_tokens: list[str],
         k1: float = 1.5, b: float = 0.75) -> tuple[np.ndarray, Counter, int]:
    """BM25 scores of every doc for the query, plus (df, N) for the gate."""
    N = len(doc_tokens)
    df: Counter = Counter()
    for d in doc_tokens:
        df.update(set(d))
    avgdl = sum(len(d) for d in doc_tokens) / max(N, 1)
    scores = np.zeros(N, np.float32)
    qset = set(q_tokens)
    for i, d in enumerate(doc_tokens):
        tf = Counter(d)
        s = 0.0
        for w in qset:
            n = tf.get(w, 0)
            if not n:
                continue
            idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
            s += idf * n * (k1 + 1) / (n + k1 * (1 - b + b * len(d) / avgdl))
        scores[i] = s
    return scores, df, N


def gate_of(q_tokens: list[str], df: Counter, N: int) -> float:
    """Jurisdiction of the lexical view: 0 unless the query contains a
    corpus-rare term, ramping to 1 as the rarest term's IDF grows."""
    if not q_tokens or N == 0:
        return 0.0
    max_idf = max(math.log(1 + (N - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5))
                  for w in q_tokens)
    return float(min(1.0, max(0.0, (max_idf - GATE_FLOOR) / GATE_SPAN)))


def rrf(primary: np.ndarray, secondary: np.ndarray, weight: float,
        k: int = RRF_K) -> np.ndarray:
    """Reciprocal-rank fusion: combines incomparable score scales by rank.
    Ranks are only meaningful within ONE candidate pool — never fuse
    per-store and concatenate. weight=0 preserves the primary ordering.
    A zero secondary score is NO evidence, not a rank — only candidates the
    secondary view actually scored receive its contribution."""
    n = len(primary)
    order = np.argsort(-primary)
    ranks = np.empty(n, np.int64)
    ranks[order] = np.arange(n)
    out = 1.0 / (k + ranks.astype(np.float32))
    if weight > 0.0:
        pos = secondary > 0
        if pos.any():
            s_order = np.argsort(-np.where(pos, secondary, -np.inf))
            s_ranks = np.empty(n, np.int64)
            s_ranks[s_order] = np.arange(n)
            out += np.where(pos, weight / (k + s_ranks), 0.0).astype(np.float32)
    return out
