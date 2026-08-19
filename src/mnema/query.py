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
from .vaults import load_vaults, refresh_vault, sync_vault
from .views import Views, postings_of

# Nearest-address support bands (dense-cosine scale; calibrated on real
# corpora — "sparse" honestly means "adjacent ground, no exact address").
SUPPORT_SETTLED = 0.65
SUPPORT_UNWRITTEN = 0.55

# Relate hop: each of the top RELATE_HITS hits' stored value vectors probes
# every OTHER consulted pool. Claim-to-claim cosines run hot (0.65–0.80 is
# ordinary), so these are relatedness weights — never support-verdict material.
# RELATE_FLOOR admits on RAW cosine (the scale users know); admitted rows are
# RANKED hub-penalized — centered cosine (the union mean direction removed)
# minus each row's self-hubness, the mean of its RELATE_HUB_K nearest centered
# cosines within its own store — so summary chunks near everything in their
# own document stop crowding out sharp connections.
RELATE_FLOOR = 0.68
RELATE_HITS = 3                      # hits that take the hop, in rank order
RELATE_PER_STORE = 2                 # rows per non-origin pool per hit
RELATE_TOTAL = 6                     # rows across pools per hit
RELATE_MAX = 12                      # rows per ask across all hits
RELATE_HUB_K = 10

VAULT_MATCH_KEYS = ("model", "dim", "seed", "sigma", "d_embed", "beta")

# Library-scale gates. Below PRE_MIN rows every code path is the exact one —
# small stores keep today's behavior bit-for-bit. Above it, resolution runs
# coarse-then-exact: the 256-dim sidecar (a seeded projection of the SAME
# lifted rows, cosine-preserving) nominates PRE_CAND candidates, which are
# rescored EXACTLY in full dimension — so scores shown are always true
# cosines, and only the candidate cut is approximate.
PRE_MIN = 100_000
PRE_CAND = 5_000
SCAN_BLOCK = 65_536
RELATE_SLACK = 0.10                  # coarse margin under RELATE_FLOOR


def _matvec(M, q: np.ndarray, block: int = SCAN_BLOCK) -> np.ndarray:
    """M @ q for an fp16 memmap (or any array), blockwise: fp32 math, flat
    memory, never materializes M."""
    n = M.shape[0]
    out = np.empty(n, np.float32)
    for a in range(0, n, block):
        out[a:a + block] = np.asarray(M[a:a + block], np.float32) @ q
    return out


def _rows32(M, idx) -> np.ndarray:
    return np.asarray(M[np.asarray(idx)], np.float32)


class _RowReader:
    """entries[i] for a fully caught-up store without parsing the log: rows
    are located by the byte-offset sidecar and parsed on first touch — an ask
    reads the handful of rows it actually shows."""

    def __init__(self, store: Store):
        self.store = store
        self._n = store.entry_count()
        self._cache: dict[int, object] = {}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i: int):
        e = self._cache.get(i)
        if e is None:
            e = self._cache[i] = self.store.read_rows([i])[0]
        return e


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
    cos: float                       # raw cosine: relatedness weight, not support
    text: str
    score: float = 0.0               # hub-penalized ranking score (diagnostics)


@dataclass
class Answer:
    support: float
    verdict: str                     # "settled" | "sparse" | "unwritten"
    hits: list[Hit]
    related_by_hit: list[list[Related]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def related(self) -> list[Related]:
        """The top hit's relations (hits at rank RELATE_HITS+ carry none)."""
        return self.related_by_hit[0] if self.related_by_hit else []


@dataclass
class _Pool:
    source: str
    entries: list
    res: np.ndarray                  # resolution cosines (masked -> -inf)
    currency: np.ndarray
    V: np.ndarray                    # fp16 value rows (memmap); gather, don't scan
    docs: list[list[str]] | None     # tokenized docs (re-derived pools only)
    superseded_by: dict[int, str]
    displaced: dict[str, float]
    views: Views | None = None       # the fold's read-side views, when current
    ksum: np.ndarray | None = None   # the fold's key sum and count: the pool's
    n: int = 0                       # share of the union mean direction
    Vc: np.ndarray | None = None     # centered live rows, derived per ask on demand
    store: Store | None = None       # for the prefilter sidecar at scale


def _views_of(store: Store) -> Views | None:
    return getattr(store, "views", None)


def _currency_of(st: core.FoldState, K, V, n: int,
                 clusters: list[list[int]]) -> np.ndarray:
    currency = np.ones(n, np.float32)
    rows = sorted({j for cl in clusters for j in cl})
    if rows:
        Kr = _rows32(K, rows)                # gather only the clustered rows
        Vr = _rows32(V, rows)
        agree = np.clip(np.einsum("ij,ij->i", Kr @ st.S.T, Vr), 1e-6, None)
        a_of = dict(zip(rows, agree))
        for cl in clusters:
            mx = max(a_of[j] for j in cl)
            for j in cl:
                currency[j] = min(currency[j], np.float32(a_of[j] / mx))
    return currency


def _lex_rows(vw: Views | None, q_terms: set[str] | None) -> np.ndarray | None:
    """Rows posting any RARE query term — the lexical arm's jurisdiction.
    These join the rescore candidates so a row that matters only lexically
    (rare jargon in a dense-dissimilar chunk) still carries its true dense
    cosine into rank fusion. Common terms are skipped: their BM25 weight is
    negligible and their posting lists are the whole corpus."""
    if vw is None or not q_terms:
        return None
    rows = []
    for w in q_terms:
        r, _ = postings_of(vw, w)
        if 0 < len(r) <= 4 * PRE_CAND:
            rows.append(r)
    return np.concatenate(rows).astype(np.int64) if rows else None


def _resolution(store: Store, K, V, n: int, q: np.ndarray,
                extra: np.ndarray | None = None) -> np.ndarray:
    """max(cos(q,key), cos(q,value)) per row. Exact full scan below PRE_MIN;
    above it, the prefilter sidecar nominates PRE_CAND rows per arm — plus
    the lexical rows in `extra` — and the candidates are rescored EXACTLY,
    so every cosine shown or fused is a true one. Non-candidates read -inf:
    dense ranks above the candidate horizon are preserved exactly."""
    pre = store.prefilter(n) if n >= PRE_MIN else None
    if pre is None:
        return np.maximum(_matvec(K[:n], q), _matvec(V[:n], q))
    preK, preV, P = pre
    qp = q @ P
    qn = float(np.linalg.norm(qp))
    qp = (qp / qn).astype(np.float32) if qn else qp.astype(np.float32)
    cands = []
    for preM in (preK, preV):
        coarse = _matvec(preM[:n], qp)
        cands.append(np.argpartition(-coarse, min(PRE_CAND, n - 1))[:PRE_CAND])
    if extra is not None and len(extra):
        cands.append(extra)
    cand = np.unique(np.concatenate(cands))
    cand = cand[cand < n]
    res = np.full(n, -np.inf, np.float32)
    res[cand] = np.maximum(_rows32(K, cand) @ q, _rows32(V, cand) @ q)
    return res


def _pool_from_views(st: core.FoldState, q: np.ndarray, source: str,
                     entries, V, K, vw: Views, store: Store,
                     q_terms: set[str] | None = None) -> _Pool:
    n = vw.n
    res = _resolution(store, K, V, n, q, extra=_lex_rows(vw, q_terms))
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
                 vw.displaced, views=vw, ksum=st.ksum, n=st.n, store=store)


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
                entries: list | None = None,
                q_terms: set[str] | None = None) -> _Pool:
    all_entries = entries if entries is not None else store.entries()
    V = store.vectors(VEC_V)                 # fp16 memmaps: rows are gathered
    K = store.vectors(VEC_K)                 # or scanned blockwise, never
    n = min(len(all_entries), V.shape[0], K.shape[0])   # materialized whole
    entries = all_entries[:n]

    # The persisted views ARE this derivation (the law: cached == recomputed);
    # re-derive below only when the ask masks rows the fold cannot know about
    # (--as-of, --current) or the views lag the log (an unwritable mirror
    # whose vectors trail its entries).
    vw = _views_of(store)
    if (vw is not None and as_of is None and not current_only
            and vw.n == n == len(all_entries)):
        return _pool_from_views(st, q, source, entries, V, K, vw, store, q_terms)

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

    res = _resolution(store, K, V, n, q)
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
                 displaced, ksum=st.ksum, n=st.n, store=store)


def _mean_direction(pools: list[_Pool]) -> np.ndarray | None:
    """The union mean direction of everything folded into the consulted pools,
    from each fold's key sum and count — no scan. None when nothing is folded."""
    n = sum(p.n for p in pools)
    if n == 0:
        return None
    mu = sum((p.ksum for p in pools if p.ksum is not None),
             np.zeros(pools[0].V.shape[1], np.float32))
    norm = float(np.linalg.norm(mu))
    return None if norm == 0.0 else (mu / norm).astype(np.float32)


def _center(X: np.ndarray, mu: np.ndarray | None) -> np.ndarray:
    """Remove the projection on the mean direction and renormalize rows."""
    if mu is None:
        return X
    Y = X - np.outer(X @ mu, mu)
    norm = np.linalg.norm(Y, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return Y / norm


def _centered_live(pool: _Pool, mu: np.ndarray | None) -> np.ndarray:
    if pool.Vc is None:
        live = np.flatnonzero(np.isfinite(pool.res))
        pool.Vc = _center(_rows32(pool.V, live), mu)
    return pool.Vc


def _raw_relate(pool: _Pool, v: np.ndarray) -> np.ndarray:
    """Raw cosines of one anchor value against a pool's rows. Exact scan below
    PRE_MIN; above it the value-side prefilter nominates every row whose
    coarse cosine clears RELATE_FLOOR - RELATE_SLACK, and those are rescored
    exactly — the floor test below always sees true cosines."""
    n = pool.V.shape[0]
    pre = pool.store.prefilter(n) if (pool.store is not None
                                      and n >= PRE_MIN) else None
    if pre is None:
        raw = _matvec(pool.V, v)
    else:
        _, preV, P = pre
        vp = v @ P
        norm = float(np.linalg.norm(vp))
        vp = (vp / norm).astype(np.float32) if norm else vp.astype(np.float32)
        coarse = _matvec(preV[:n], vp)
        keep = np.flatnonzero(coarse >= RELATE_FLOOR - RELATE_SLACK)
        if len(keep) > 4 * PRE_CAND:
            keep = keep[np.argpartition(-coarse[keep], 4 * PRE_CAND)[:4 * PRE_CAND]]
        raw = np.full(n, -np.inf, np.float32)
        if len(keep):
            keep = np.sort(keep)
            raw[keep] = _rows32(pool.V, keep) @ v
    raw[~np.isfinite(pool.res)] = -np.inf
    return raw


def _self_hubness(pool: _Pool, cand: np.ndarray, Yc: np.ndarray,
                  mu: np.ndarray | None) -> np.ndarray:
    """r(y) per candidate: the mean of its RELATE_HUB_K largest centered cosines
    to the OTHER live rows of its own pool (all of them when fewer)."""
    live_rows = np.flatnonzero(np.isfinite(pool.res))
    Vc = _centered_live(pool, mu)
    if len(live_rows) < 2:
        return np.zeros(len(cand), np.float32)
    C = Vc @ Yc.T                                        # (live, cand)
    C[np.searchsorted(live_rows, cand), np.arange(len(cand))] = -np.inf
    k = min(RELATE_HUB_K, len(live_rows) - 1)
    top = np.partition(C, -k, axis=0)[-k:]
    return top.mean(axis=0)


def _relate_one(pools: list[_Pool], origin: _Pool, row: int,
                mu: np.ndarray | None, exclude: set[str], seen: set[str],
                budget: int) -> list[Related]:
    """One hit's hop: its stored value row (unit, so a dot IS the cosine)
    probes every OTHER pool — origin excluded, so an answer never echoes its
    own store. Rows at or above RELATE_FLOOR (raw) are ranked by
    2·cos_c − r(y); RELATE_PER_STORE per pool, RELATE_TOTAL across pools,
    deduped by topic (keyless rows by hash) and against `seen`."""
    v = np.asarray(origin.V[row], np.float32)
    vc = _center(v[None, :], mu)[0]
    picks: list[tuple[float, Related]] = []
    for pool in pools:
        if pool.source == origin.source:
            continue
        raw = _raw_relate(pool, v)
        cand = np.flatnonzero(raw >= RELATE_FLOOR)
        cand = np.array([i for i in cand if pool.entries[int(i)].h not in exclude],
                        dtype=np.int64)
        if not len(cand):
            continue
        Yc = _center(_rows32(pool.V, cand), mu)
        score = 2.0 * (Yc @ vc) - _self_hubness(pool, cand, Yc, mu)
        taken: set[str] = set()
        for j in np.argsort(-score, kind="stable"):
            if len(taken) == RELATE_PER_STORE:
                break
            e = pool.entries[int(cand[j])]
            key = e.topic or e.h
            if key in taken or key in seen:
                continue
            taken.add(key)
            picks.append((float(score[j]),
                          Related(pool.source, e.topic, e.at, e.kind, e.h,
                                  float(raw[cand[j]]), e.text, float(score[j]))))
    picks.sort(key=lambda p: -p[0])
    out = [r for _, r in picks[:min(RELATE_TOTAL, budget)]]
    seen.update(r.topic or r.h for r in out)
    return out


def _relate(pools: list[_Pool], anchors: list[tuple[_Pool, int]]
            ) -> list[list[Related]]:
    """The relate hop for the top RELATE_HITS hits, in rank order: relations
    already shown under an earlier hit are not repeated, no hit relates to
    another shown hit, and RELATE_MAX bounds the whole ask."""
    mu = _mean_direction(pools)
    shown = {pool.entries[i].h for pool, i in anchors}
    seen: set[str] = set()
    out: list[list[Related]] = []
    total = 0
    for pool, i in anchors[:RELATE_HITS]:
        rel = _relate_one(pools, pool, i, mu, shown, seen, RELATE_MAX - total)
        total += len(rel)
        out.append(rel)
    return out


def _try_fast_pool(store: Store, q: np.ndarray, source: str,
                   q_terms: set[str] | None = None) -> _Pool | None:
    """Entries-free pool for a FULLY caught-up store: log line count, vector
    rows, folded state, and views all agree, so catch_up would be a no-op and
    no ask feature needs the parsed log. Any disagreement returns None and
    the caller takes today's exact path — staleness is never served."""
    try:
        n_log = store.entry_count()
    except OSError:
        return None
    n_vec = min(store._rows_on_disk(VEC_V), store._rows_on_disk(VEC_K))
    if n_log == 0 or n_vec != n_log:
        return None
    st = store.load_state()
    if st.n != n_vec:
        return None
    vw = store.load_views()
    if vw.n != n_vec:
        return None
    store.views = vw
    return _pool_from_views(st, q, source, _RowReader(store),
                            store.vectors(VEC_V), store.vectors(VEC_K), vw,
                            store, q_terms)


def ask(store: Store, question: str, top: int = 5, as_of: str | None = None,
        current_only: bool = False, slots: dict[str, str] | None = None,
        embedder: Embedder | None = None, vaults: str = "all",
        exclude: tuple[str, ...] = ()) -> Answer:
    """vaults: "all" (local + every vault), "local", or a vault name.
    exclude: vault names left out of BOTH hops — no pool is built for them."""
    embedder = embedder or Embedder(store.cfg["model"])
    if exclude:
        known = {v["name"] for v in load_vaults(store.path)}
        for name in exclude:
            if name not in known:
                raise SystemExit(f"no vault named '{name}' (see: mnema vault list)")

    q_emb = embedder.queries([question])
    value_lift, key_lift = store._lift_ops()
    q = value_lift(q_emb)[0] if not store.cfg["sigma"] else key_lift(q_emb)[0]
    if slots:
        q = core.bind_slots(q, slots, store.cfg["seed"])
    q_tokens = tokenize(question)
    q_terms = set(q_tokens)

    pools: list[_Pool] = []
    warnings: list[str] = []

    if vaults in ("all", "local"):
        fast = (None if (as_of or current_only)
                else _try_fast_pool(store, q, "local", q_terms))
        if fast is not None:
            pools.append(fast)
        else:
            entries = store.entries()
            st = store.catch_up(embedder, quiet=True, entries=entries)
            if as_of:
                st = store.fold_prefix(as_of, entries=entries)
            pools.append(_build_pool(store, st, q, as_of, current_only, "local",
                                     entries=entries, q_terms=q_terms))

    if vaults != "local":
        wanted = None if vaults == "all" else vaults
        matched = False
        for vault in load_vaults(store.path):
            if wanted and vault["name"] != wanted:
                continue
            if vault["name"] in exclude:
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
                # the publisher may have recompiled onto a new model: same
                # address, same log bytes, NEW config — re-check the wire
                refreshed = refresh_vault(store.path, vault)
                if refreshed is not None:
                    vs = Store(refreshed)
                    mismatch = [k for k in VAULT_MATCH_KEYS
                                if vs.cfg.get(k) != store.cfg.get(k)]
            if mismatch:
                warnings.append(f"vault '{vault['name']}' skipped: "
                                f"config differs on {mismatch}")
                continue
            fast = (None if (as_of or current_only)
                    else _try_fast_pool(vs, q, vault["name"], q_terms))
            if fast is not None:
                pools.append(fast)
                continue
            vault_entries = vs.entries()
            vst = vs.catch_up(quiet=True, embed_missing=False, entries=vault_entries)
            if as_of:
                vst = vs.fold_prefix(as_of, entries=vault_entries)
            pools.append(_build_pool(vs, vst, q, as_of, current_only,
                                     vault["name"], entries=vault_entries,
                                     q_terms=q_terms))
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
    anchors: list[tuple[_Pool, int]] = []
    for gi in np.argsort(-final)[:top]:
        if not np.isfinite(final[gi]):
            continue
        pi = int(np.searchsorted(offsets, gi, side="right") - 1)
        pool = pools[pi]
        i = int(gi - offsets[pi])
        anchors.append((pool, i))
        e = pool.entries[i]
        hits.append(Hit(float(res_all[gi]), i, e.at, e.kind, e.topic, e.text,
                        pool.superseded_by.get(i), h=e.h,
                        displaced=pool.displaced.get(e.h), source=pool.source))
    related = _relate(pools, anchors) if len(pools) > 1 else []
    return Answer(round(sup, 2), verdict, hits, related, warnings)
