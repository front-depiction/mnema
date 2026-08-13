"""Asking questions: factored retrieval across the local store and any vaults.

The score is a product of factors, each with authority over ONE concern and a
jurisdiction it is consulted in (see README "The mathematics"):

    resolution  is it about this?      max(cos(q,key), cos(q,value)) — dense,
                                       fused with gated lexical evidence
    currency    does it still stand?   <S k, v> normalized within the entry's
                                       declared supersession family, per store
    support     is anything here?      max resolution -> verdict bands

Cross-pool semantics: dense cosines share one scale (configs must match) and
concatenate across stores; RANKS do not — lexical/BM25 statistics, the IDF
gate, and rank fusion are computed once over the UNION candidate pool.
Currency stays store-local: supersession is a fact about one store's history."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import core
from .embed import Embedder
from .lexical import bm25_index, gate_of, rrf, tokenize
from .store import VEC_K, VEC_V, Store
from .vaults import load_vaults, sync_vault
from .views import Views, postings_of

# Nearest-address support bands (dense-cosine scale; calibrated on real
# corpora — "sparse" honestly means "adjacent ground, no exact address").
SUPPORT_SETTLED = 0.65
SUPPORT_UNWRITTEN = 0.55

# Relate hop: the top hit's stored value vector probes every OTHER consulted
# pool. Claim-to-claim cosines run hot (0.65–0.80 is ordinary), so these are
# relatedness weights — never support-verdict material.
RELATE_FLOOR = 0.68
RELATE_TOP = 3                       # rows surfaced per non-origin pool

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
class Related:
    source: str
    topic: str | None
    at: str
    kind: str
    h: str
    cos: float                       # relatedness weight, not a support score
    text: str


@dataclass
class Answer:
    support: float
    verdict: str                     # "settled" | "sparse" | "unwritten"
    hits: list[Hit]
    related: list[Related] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Pool:
    source: str
    entries: list
    res: np.ndarray                  # resolution cosines (masked -> -inf)
    currency: np.ndarray
    V: np.ndarray                    # float32 value rows, one per entry
    docs: list[list[str]] | None     # tokenized docs (re-derived pools only)
    superseded_by: dict[int, str]
    displaced: dict[str, float]
    views: Views | None = None       # the fold's read-side views, when current


def _views_of(store: Store) -> Views | None:
    return getattr(store, "views", None)


def _currency_of(st: core.FoldState, K: np.ndarray, V: np.ndarray, n: int,
                 clusters: list[list[int]]) -> np.ndarray:
    currency = np.ones(n, np.float32)
    rows = sorted({j for cl in clusters for j in cl})
    if rows:
        Kr = K[rows]
        agree = np.clip(np.einsum("ij,ij->i", Kr @ st.S.T, V[rows]), 1e-6, None)
        a_of = dict(zip(rows, agree))
        for cl in clusters:
            mx = max(a_of[j] for j in cl)
            for j in cl:
                currency[j] = min(currency[j], np.float32(a_of[j] / mx))
    return currency


def _pool_from_views(st: core.FoldState, q: np.ndarray, source: str,
                     entries: list, V: np.ndarray, K: np.ndarray,
                     vw: Views) -> _Pool:
    n = vw.n
    res = np.maximum(K[:n] @ q, V[:n] @ q)
    for ai, pi in vw.alias_parent:
        res[pi] = max(res[pi], res[ai])  # extra address for the same belief
    res[~vw.mask] = -np.inf

    clusters = [g for g in vw.topic_groups.values() if len(g) > 1]
    clusters += [list(pair) for pair in vw.disp_pairs]
    currency = _currency_of(st, K, V, n, clusters)

    superseded_by = {}
    for g in vw.topic_groups.values():
        if len(g) > 1:
            at = entries[g[-1]].at
            for i in g[:-1]:
                superseded_by[i] = at
    return _Pool(source, entries, res, currency, V[:n], None, superseded_by,
                 vw.displaced, views=vw)


def _pool_doc_len(pool: _Pool) -> np.ndarray:
    if pool.views is not None:
        return pool.views.doc_len
    return np.fromiter((len(d) for d in pool.docs), np.int32, len(pool.docs))


def _pool_postings(pool: _Pool, q_terms: set[str]) -> dict:
    """The query terms' (rows, tf) posting lists over this pool's rows — read
    straight off the views' inverted index when current, else derived from
    the ask-masked tokenized docs (--as-of/--current asks, lagging mirrors)."""
    if pool.views is not None:
        return {w: postings_of(pool.views, w) for w in q_terms}
    posts: dict = {w: ([], []) for w in q_terms}
    for i, d in enumerate(pool.docs):
        if not d:
            continue
        tf = Counter(d)
        for w in q_terms:
            c = tf.get(w)
            if c:
                posts[w][0].append(i)
                posts[w][1].append(c)
    return {w: (np.asarray(rs, np.int32), np.asarray(ts, np.int32))
            for w, (rs, ts) in posts.items()}


def _build_pool(store: Store, st: core.FoldState, q: np.ndarray,
                as_of: str | None, current_only: bool, source: str,
                entries: list | None = None) -> _Pool:
    all_entries = entries if entries is not None else store.entries()
    V = store.vectors(VEC_V).astype(np.float32)
    K = store.vectors(VEC_K).astype(np.float32)
    n = min(len(all_entries), V.shape[0], K.shape[0])
    entries = all_entries[:n]

    # The persisted views ARE this derivation (the law: cached == recomputed);
    # re-derive below only when the ask masks rows the fold cannot know about
    # (--as-of, --current) or the views lag the log (an unwritable mirror
    # whose vectors trail its entries).
    vw = _views_of(store)
    if (vw is not None and as_of is None and not current_only
            and vw.n == n == len(all_entries)):
        return _pool_from_views(st, q, source, entries, V, K, vw)

    retracted = store.retracted_hashes(entries)
    mask = np.array([(e.at <= as_of if as_of else True)
                     and e.kind not in ("keep", "retract", "alias")
                     and e.h not in retracted
                     for e in entries], dtype=bool)

    last = store.latest_by_topic(all_entries)
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

    # Currency competition is defined by the log's own supersession relations —
    # same-topic groups and displacement edges — never by N^2 semantic
    # clustering. Unrelated entries hold currency 1 by definition; the state's
    # testimony <S k, v> is computed only for the few entries actually in a
    # replacement relationship. Query cost: linear scans only.
    revoked = {e.target for e in entries if e.kind == "keep" and e.target}
    clusters: list[list[int]] = []
    by_topic: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        if e.kind in ("keep", "retract", "alias") or e.h in retracted:
            continue
        if e.topic:
            by_topic.setdefault(e.topic, []).append(i)
    clusters.extend(ix for ix in by_topic.values() if len(ix) > 1)
    for i, e in enumerate(entries):
        if e.h in retracted:
            continue
        for target_h, _w in e.displaces:
            if (target_h in row_of and target_h not in revoked
                    and target_h not in retracted):   # a forgotten target must
                clusters.append([i, row_of[target_h]])  # not depress a live rival

    currency = _currency_of(st, K, V, n, clusters)

    superseded_by = {}
    for i, e in enumerate(entries):
        if e.topic and last.get(e.topic) not in (None, i) and not as_of:
            superseded_by[i] = entries[last[e.topic]].at

    displaced: dict[str, float] = {}
    for e in entries:
        if e.h in retracted:
            continue
        for target_h, w in e.displaces:
            if target_h not in revoked:
                displaced[target_h] = max(displaced.get(target_h, 0.0), w)

    docs = [tokenize((e.topic or "") + " " + e.text) if mask[i] else []
            for i, e in enumerate(entries)]
    return _Pool(source, entries, res, currency, V[:n], docs, superseded_by,
                 displaced)


def _relate(pools: list[_Pool], origin: _Pool, row: int) -> list[Related]:
    """The relate hop: the top hit's stored value row (unit vector, so a dot
    IS the cosine) probes every OTHER consulted pool — origin excluded, so an
    answer can never echo its own store. Up to RELATE_TOP live rows per pool
    at or above RELATE_FLOOR, deduped by topic (keyless rows by hash)."""
    v = origin.V[row]
    out: list[Related] = []
    for pool in pools:
        if pool.source == origin.source:
            continue
        cos = pool.V @ v
        cos[~np.isfinite(pool.res)] = -np.inf
        taken: set[str] = set()
        for i in np.argsort(-cos):
            if cos[i] < RELATE_FLOOR or len(taken) == RELATE_TOP:
                break
            e = pool.entries[int(i)]
            key = e.topic or e.h
            if key in taken:
                continue
            taken.add(key)
            out.append(Related(pool.source, e.topic, e.at, e.kind, e.h,
                               float(cos[i]), e.text))
    return out


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
        entries = store.entries()
        st = store.catch_up(embedder, quiet=True, entries=entries)
        if as_of:
            st = store.fold_prefix(as_of, entries=entries)
        pools.append(_build_pool(store, st, q, as_of, current_only, "local",
                                 entries=entries))

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
            vault_entries = vs.entries()
            vst = vs.catch_up(quiet=True, embed_missing=False, entries=vault_entries)
            if as_of:
                vst = vs.fold_prefix(as_of, entries=vault_entries)
            pools.append(_build_pool(vs, vst, q, as_of, current_only,
                                     vault["name"], entries=vault_entries))
        if wanted and not matched:
            raise SystemExit(f"no vault named '{wanted}' (see: mnema vault list)")

    if not pools or not any(np.isfinite(p.res).any() for p in pools):
        return Answer(0.0, "unwritten", [], warnings=warnings)

    # ---- pool-global composition over the UNION of candidates
    res_all = np.concatenate([p.res for p in pools])
    cur_all = np.concatenate([p.currency for p in pools])
    finite = np.isfinite(res_all)
    offsets = np.cumsum([0] + [len(p.entries) for p in pools])

    # Union BM25/IDF from per-pool posting lists: each store's views (or its
    # re-derived docs) contribute the query terms' postings; df, N, and avgdl
    # merge across stores, so rarity is judged against everything visible —
    # the same statistics the corpus scan over concatenated docs produced.
    q_tokens = tokenize(question)
    q_terms = set(q_tokens)
    postings = {w: ([], []) for w in q_terms}
    for pool, off in zip(pools, offsets):
        for w, (rs, ts) in _pool_postings(pool, q_terms).items():
            if len(rs):
                postings[w][0].append(np.asarray(rs, np.int64) + off)
                postings[w][1].append(np.asarray(ts))
    merged = {w: (np.concatenate(rs) if rs else np.empty(0, np.int64),
                  np.concatenate(ts) if ts else np.empty(0, np.int64))
              for w, (rs, ts) in postings.items()}
    doc_len_all = np.concatenate([_pool_doc_len(p) for p in pools])
    lex, df, n_docs = bm25_index(merged, doc_len_all, q_tokens)
    gate = gate_of(q_tokens, df, n_docs)
    lex[~finite] = -np.inf

    fused = rrf(res_all, lex, weight=gate)
    final = fused * cur_all
    final[~finite] = -np.inf

    sup = float(res_all[finite].max())
    verdict = ("settled" if sup >= SUPPORT_SETTLED
               else "unwritten" if sup < SUPPORT_UNWRITTEN else "sparse")

    hits: list[Hit] = []
    top_hit: tuple[_Pool, int] | None = None
    for gi in np.argsort(-final)[:top]:
        if not np.isfinite(final[gi]):
            continue
        pi = int(np.searchsorted(offsets, gi, side="right") - 1)
        pool = pools[pi]
        i = int(gi - offsets[pi])
        if top_hit is None:
            top_hit = (pool, i)
        e = pool.entries[i]
        hits.append(Hit(float(res_all[gi]), i, e.at, e.kind, e.topic, e.text,
                        pool.superseded_by.get(i), h=e.h,
                        displaced=pool.displaced.get(e.h), source=pool.source))
    related = (_relate(pools, *top_hit)
               if top_hit is not None and len(pools) > 1 else [])
    return Answer(round(sup, 2), verdict, hits, related, warnings)
