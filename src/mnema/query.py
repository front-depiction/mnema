"""Asking questions: factored retrieval across the local store and any vaults.

The score is a product of factors, each with authority over ONE concern and a
jurisdiction it is consulted in (see README "The mathematics"):

    resolution  is it about this?      max(cos(q,key), cos(q,value)) — dense,
                                       fused with gated lexical evidence
    currency    does it still stand?   <S k, v> normalized within the entry's
                                       own address cluster, per store
    support     is anything here?      max resolution -> verdict bands

Cross-pool semantics: dense cosines share one scale (configs must match) and
concatenate across stores; RANKS do not — lexical/BM25 statistics, the IDF
gate, and rank fusion are computed once over the UNION candidate pool.
Currency stays store-local: supersession is a fact about one store's history."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import core
from .embed import Embedder
from .lexical import bm25, gate_of, rrf, tokenize
from .store import VEC_K, VEC_V, Store
from .vaults import load_vaults, sync_vault

# Nearest-address support bands (dense-cosine scale; calibrated on real
# corpora — "sparse" honestly means "adjacent ground, no exact address").
SUPPORT_SETTLED = 0.65
SUPPORT_UNWRITTEN = 0.55
CLUSTER = 0.90

VAULT_MATCH_KEYS = ("model", "dim", "seed", "sigma", "d_embed", "beta")


@dataclass
class Hit:
    score: float                     # resolution cosine (display scale)
    index: int
    at: str
    kind: str
    topic: str | None
    text: str
    superseded_by: str | None
    h: str = ""
    displaced: float | None = None
    source: str = "local"


@dataclass
class Answer:
    support: float
    verdict: str                     # "settled" | "sparse" | "unwritten"
    hits: list[Hit]
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Pool:
    source: str
    entries: list
    res: np.ndarray                  # resolution cosines (masked -> -inf)
    currency: np.ndarray
    docs: list[list[str]]
    superseded_by: dict[int, str]
    displaced: dict[str, float]


def _build_pool(store: Store, st: core.FoldState, q: np.ndarray,
                as_of: str | None, current_only: bool, source: str) -> _Pool:
    entries = store.entries()
    V = store.vectors(VEC_V).astype(np.float32)
    K = store.vectors(VEC_K).astype(np.float32)
    n = min(len(entries), V.shape[0], K.shape[0])
    entries = entries[:n]

    retracted = store.retracted_hashes(entries)
    mask = np.array([(e.at <= as_of if as_of else True)
                     and e.kind not in ("keep", "retract", "alias")
                     and e.h not in retracted
                     for e in entries], dtype=bool)

    last = store.latest_by_topic()
    if current_only:
        current_idx = set(last.values())
        for i in range(n):
            if i not in current_idx:
                mask[i] = False

    res = np.maximum(K[:n] @ q, V[:n] @ q)
    row_of = {e.h: i for i, e in enumerate(entries)}
    for i, e in enumerate(entries):
        if e.kind == "alias" and e.target in row_of:
            pi = row_of[e.target]
            res[pi] = max(res[pi], res[i])   # extra address for the same belief
    res[~mask] = -np.inf

    agreement = np.clip(np.einsum("ij,ij->i", K[:n] @ st.S.T, V[:n]), 1e-6, None)
    sim = K[:n] @ K[:n].T
    currency = np.empty(n, np.float32)
    for i in range(n):
        cluster = sim[i] >= CLUSTER
        cluster[i] = True        # zero-vector bookkeeping rows: self-cluster
        currency[i] = agreement[i] / agreement[cluster].max()

    superseded_by = {}
    for i, e in enumerate(entries):
        if e.topic and last.get(e.topic) not in (None, i) and not as_of:
            superseded_by[i] = entries[last[e.topic]].at

    revoked = {e.target for e in entries if e.kind == "keep" and e.target}
    displaced: dict[str, float] = {}
    for e in entries:
        if e.h in retracted:
            continue
        for target_h, w in e.displaces:
            if target_h not in revoked:
                displaced[target_h] = max(displaced.get(target_h, 0.0), w)

    docs = [tokenize((e.topic or "") + " " + e.text) if mask[i] else []
            for i, e in enumerate(entries)]
    return _Pool(source, entries, res, currency, docs, superseded_by, displaced)


def ask(store: Store, question: str, top: int = 5, as_of: str | None = None,
        current_only: bool = False, slots: dict[str, str] | None = None,
        embedder: Embedder | None = None, vaults: str = "all") -> Answer:
    """vaults: "all" (local + every vault), "local", or a vault name."""
    embedder = embedder or Embedder(store.cfg["model"])

    q_emb = embedder.queries([question])
    value_lift, key_lift = store._lift_ops()
    q = value_lift(q_emb)[0] if not store.cfg["sigma"] else key_lift(q_emb)[0]
    if slots:
        q = core.bind_slots(q, slots, store.cfg["seed"])

    pools: list[_Pool] = []
    warnings: list[str] = []

    if vaults in ("all", "local"):
        st = store.catch_up(embedder, quiet=True)
        if as_of:
            st = store.fold_prefix(as_of)
        pools.append(_build_pool(store, st, q, as_of, current_only, "local"))

    if vaults != "local":
        wanted = None if vaults == "all" else vaults
        matched = False
        for vault in load_vaults(store.path):
            if wanted and vault["name"] != wanted:
                continue
            matched = True
            try:
                vs = Store(sync_vault(store.path, vault))
            except Exception as ex:
                warnings.append(f"vault '{vault['name']}' unreachable: {ex}")
                continue
            mismatch = [k for k in VAULT_MATCH_KEYS
                        if vs.cfg.get(k) != store.cfg.get(k)]
            if mismatch:
                warnings.append(f"vault '{vault['name']}' skipped: "
                                f"config differs on {mismatch}")
                continue
            vst = vs.catch_up(quiet=True, embed_missing=False)
            if as_of:
                vst = vs.fold_prefix(as_of)
            pools.append(_build_pool(vs, vst, q, as_of, current_only,
                                     vault["name"]))
        if wanted and not matched:
            raise SystemExit(f"no vault named '{wanted}' (see: mnema vault list)")

    if not pools or not any(np.isfinite(p.res).any() for p in pools):
        return Answer(0.0, "unwritten", [], warnings)

    # ---- pool-global composition over the UNION of candidates
    res_all = np.concatenate([p.res for p in pools])
    cur_all = np.concatenate([p.currency for p in pools])
    docs_all = [d for p in pools for d in p.docs]
    finite = np.isfinite(res_all)

    lex, df, n_docs = bm25(docs_all, tokenize(question))
    gate = gate_of(tokenize(question), df, n_docs)
    lex[~finite] = -np.inf

    fused = rrf(res_all, lex, weight=gate)
    final = fused * cur_all
    final[~finite] = -np.inf

    sup = float(res_all[finite].max())
    verdict = ("settled" if sup >= SUPPORT_SETTLED
               else "unwritten" if sup < SUPPORT_UNWRITTEN else "sparse")

    offsets = np.cumsum([0] + [len(p.entries) for p in pools])
    hits: list[Hit] = []
    for gi in np.argsort(-final)[:top]:
        if not np.isfinite(final[gi]):
            continue
        pi = int(np.searchsorted(offsets, gi, side="right") - 1)
        pool = pools[pi]
        i = int(gi - offsets[pi])
        e = pool.entries[i]
        hits.append(Hit(float(res_all[gi]), i, e.at, e.kind, e.topic, e.text,
                        pool.superseded_by.get(i), h=e.h,
                        displaced=pool.displaced.get(e.h), source=pool.source))
    return Answer(round(sup, 2), verdict, hits, warnings)
