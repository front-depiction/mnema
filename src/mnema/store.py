"""On-disk store: an append-only log of entries plus derived, catch-up-able state.

Layout of a store directory:

    config.json   model, dim, seed, beta, sigma  (immutable after init)
    log.jsonl     append-only entries — THE source of truth
    vec_v.f16     lifted value vectors, one fp16 row per log line (write-once)
    vec_k.f16     lifted key vectors, same
    state.npz     S, Lam (fp32) + n = number of rows folded
    views.npz     read-side views (inverted index, topic/supersession maps)
                  — same fold, same catch-up rules, same disposability

Appends are O(1): one log line + two fp16 rows + a rank-one fold. Any command
first runs catch_up(), which embeds/folds whatever the log has that the state
hasn't — so concurrent writers and cross-session use just work. The log is the
authority; vectors and state are disposable caches (delete them and any command
rebuilds)."""

from __future__ import annotations

import fcntl
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import core, views
from .embed import Embedder

CONFIG, LOG, VEC_V, VEC_K, STATE = "config.json", "log.jsonl", "vec_v.f16", "vec_k.f16", "state.npz"
VIEWS = "views.npz"

# Displacement inference (keyless entries only): a new entry attenuates prior
# entries whose keys it lands near. Floor-gated, capped BELOW 1 — inferred
# erasure must stay invertible (beta=1 destruction is reserved for declared
# identity), so a wrong inference is recoverable via a `keep` event.
INFER_FLOOR = 0.72
INFER_CAP = 0.90
INFER_TOPK = 3

# Bulk displacement inference compares each new key against every strictly-
# prior key (growing candidate pool) — same semantics as remembering one at a
# time. Blocking bounds the per-GEMM candidate matrix to BLOCK rows instead of
# one BLAS call per entry, which is what turns this into scalar-loop overhead
# on large loads.
INFER_BLOCK = 256

# The embedder reads ~512 tokens; content beyond never reaches the index.
# Writes past this horizon get a WARNING (loud beats silent) — split long
# memories into coherent entries, which also retrieves better (measured).
HORIZON_WORDS = 350


def past_horizon(text: str) -> bool:
    return len(text.split()) > HORIZON_WORDS


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time_ns() // 1_000_000) % 1000:03d}Z"


@dataclass
class Entry:
    text: str
    topic: str | None = None
    at: str = field(default_factory=now_iso)
    kind: str = "note"
    slots: dict[str, str] = field(default_factory=dict)
    displaces: list = field(default_factory=list)   # [[target_h, weight], ...] inferred at write
    target: str | None = None                       # keep/retract/alias: the referenced entry
    questions: list = field(default_factory=list)   # optional: questions this memory answers

    @property
    def h(self) -> str:
        payload = json.dumps([self.at, self.topic, self.text,
                              sorted(self.slots.items()), self.target])
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def to_line(self) -> str:
        d = {"at": self.at, "kind": self.kind, "topic": self.topic,
             "text": self.text, "h": self.h}
        if self.slots:
            d["slots"] = self.slots
        if self.displaces:
            d["displaces"] = self.displaces
        if self.target:
            d["target"] = self.target
        if self.questions:
            d["questions"] = self.questions
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_line(cls, line: str) -> "Entry":
        d = json.loads(line)
        return cls(text=d["text"], topic=d.get("topic"), at=d["at"],
                   kind=d.get("kind", "note"), slots=d.get("slots", {}),
                   displaces=d.get("displaces", []), target=d.get("target"),
                   questions=d.get("questions", []))


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        cfg_file = self.path / CONFIG
        if not cfg_file.exists():
            raise SystemExit(f"not a mnema store: {self.path} (run `mnema init` first)")
        self.cfg = json.loads(cfg_file.read_text())
        self.dim = self.cfg["dim"]

    # -------------------------------------------------------------- lifecycle

    @classmethod
    def init(cls, path: Path, model: str, dim: int, beta: float, sigma: float | None,
             seed: int, d_embed: int) -> "Store":
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        cfg_file = path / CONFIG
        if cfg_file.exists():
            raise SystemExit(f"store already exists: {path}")
        cfg = {"model": model, "dim": dim, "beta": beta, "sigma": sigma,
               "seed": seed, "d_embed": d_embed, "version": 1}
        cfg_file.write_text(json.dumps(cfg, indent=2))
        (path / LOG).touch()
        return cls(path)

    # -------------------------------------------------------------- log

    def entries(self) -> list[Entry]:
        text = (self.path / LOG).read_text()
        return [Entry.from_line(l) for l in text.splitlines() if l.strip()]

    def known_hashes(self) -> set[str]:
        return {e.h for e in self.entries()}

    def _lock(self):
        """Exclusive advisory lock for all log/vector/state mutation: the
        positional alignment between log lines and vector rows is the one
        invariant concurrent writers could corrupt."""
        f = (self.path / ".lock").open("w")
        fcntl.flock(f, fcntl.LOCK_EX)
        return f

    def append(self, new: list[Entry]) -> int:
        """Dedupe against the log and append. Returns count actually added.
        An entry's optional `questions` expand into alias entries — extra
        question-shaped ADDRESSES for the same belief, invisible at readout.
        Serialized against concurrent writers via the store lock."""
        lock = self._lock()
        try:
            return self._append_locked(new)
        finally:
            lock.close()

    def _append_locked(self, new: list[Entry]) -> int:
        seen = self.known_hashes()
        added = 0
        with (self.path / LOG).open("a") as f:
            for e in new:
                if e.h in seen:
                    continue
                seen.add(e.h)
                f.write(e.to_line() + "\n")
                added += 1
                for q in e.questions:
                    a = Entry(text=q, kind="alias", target=e.h, at=e.at)
                    if a.h in seen:
                        continue
                    seen.add(a.h)
                    f.write(a.to_line() + "\n")
        return added

    # -------------------------------------------------------------- vectors

    def _rows_on_disk(self, name: str) -> int:
        p = self.path / name
        return p.stat().st_size // (2 * self.dim) if p.exists() else 0

    def vectors(self, name: str) -> np.ndarray:
        p = self.path / name
        if not p.exists() or p.stat().st_size == 0:
            return np.empty((0, self.dim), np.float16)
        return np.memmap(p, dtype=np.float16, mode="r").reshape(-1, self.dim)

    def _append_vectors(self, name: str, rows: np.ndarray) -> None:
        with (self.path / name).open("ab") as f:
            f.write(rows.astype(np.float16).tobytes())

    def _lift_ops(self):
        R = core.lift_projection(self.cfg["d_embed"], self.dim, self.cfg["seed"])
        if self.cfg["sigma"]:
            W, b = core.rff_projection(self.cfg["d_embed"], self.dim,
                                       self.cfg["sigma"], self.cfg["seed"])
            key_lift = lambda X: core.rff(X, W, b)
        else:
            key_lift = lambda X: core.lift(X, R)
        return (lambda X: core.lift(X, R)), key_lift

    # -------------------------------------------------------------- state

    def load_state(self) -> core.FoldState:
        p = self.path / STATE
        if not p.exists():
            return core.FoldState.empty(self.dim)
        z = np.load(p)
        if "ksum" not in z:      # older state format: refold from vectors
            return core.FoldState.empty(self.dim)
        return core.FoldState(z["S"].astype(np.float32),
                              z["Lam"].astype(np.float32),
                              z["ksum"].astype(np.float32), int(z["n"]))

    def save_state(self, st: core.FoldState) -> None:
        tmp = self.path / (STATE + ".tmp.npz")
        np.savez(tmp, S=st.S, Lam=st.Lam, ksum=st.ksum, n=st.n)
        tmp.rename(self.path / STATE)

    def load_views(self) -> views.Views:
        p = self.path / VIEWS
        if not p.exists():
            return views.Views.empty()
        try:
            return views.load(p)
        except Exception:
            return views.Views.empty()    # corrupt sidecar: refold is the fallback

    def save_views(self, vw: views.Views) -> None:
        tmp = self.path / (VIEWS + ".tmp.npz")
        try:
            views.save(vw, tmp)
            tmp.rename(self.path / VIEWS)
        except OSError:
            pass    # read-only mirror: the in-memory views still serve this ask

    # -------------------------------------------------------------- catch-up

    def _embed_missing(self, embedder: Embedder | None, quiet: bool,
                       entries: list[Entry] | None = None) -> list[Entry]:
        entries = entries if entries is not None else self.entries()
        n_vec = min(self._rows_on_disk(VEC_V), self._rows_on_disk(VEC_K))
        missing = entries[n_vec:]
        if missing:
            if embedder is None:
                embedder = Embedder(self.cfg["model"])
            if not quiet:
                print(f"embedding {len(missing)} new entries...", flush=True)
            real = [e for e in missing if e.kind not in ("keep", "retract")]
            value_lift, key_lift = self._lift_ops()
            if real:
                Vr = value_lift(embedder.passages([e.text for e in real]))
                Kr = key_lift(embedder.passages([e.topic or e.text for e in real]))
            V = np.zeros((len(missing), self.dim), np.float32)
            K = np.zeros((len(missing), self.dim), np.float32)
            n_prior = len(entries) - len(missing)
            prior_row = {e.h: i for i, e in enumerate(entries[:n_prior])}
            Vd = self.vectors(VEC_V)
            j = 0
            for i, e in enumerate(missing):
                if e.kind in ("keep", "retract"):
                    continue                      # bookkeeping events hold zero rows
                K[i] = core.bind_slots(Kr[j], e.slots, self.cfg["seed"])
                if e.kind == "alias" and e.target:
                    # alias: question-shaped address, the PARENT's value
                    if e.target in prior_row and prior_row[e.target] < Vd.shape[0]:
                        V[i] = Vd[prior_row[e.target]].astype(np.float32)
                    else:
                        for bi, be in enumerate(missing[:i]):
                            if be.h == e.target:
                                V[i] = V[bi]
                                break
                else:
                    V[i] = Vr[j]
                j += 1
            self._append_vectors(VEC_V, V)
            self._append_vectors(VEC_K, K)
        return entries

    def retracted_hashes(self, entries: list[Entry] | None = None) -> set[str]:
        entries = entries if entries is not None else self.entries()
        return {e.target for e in entries if e.kind == "retract" and e.target}

    def _build_ops(self, entries: list[Entry], start: int) -> list[tuple]:
        """Compile log entries [start:] into fold ops. A `keep` event revokes
        every displacement of its target; a `retract` event removes its target
        entry from the fold entirely (write and displacements both). Both act
        on past and future occurrences, so callers refold from zero when one
        appears in the tail — reversal by re-derivation, exact regardless of
        the folded operators being non-invertible."""
        revoked = {e.target for e in entries if e.kind == "keep" and e.target}
        retracted = self.retracted_hashes(entries)
        row_of = {e.h: i for i, e in enumerate(entries)}
        ops: list[tuple] = []
        for i in range(start, len(entries)):
            e = entries[i]
            if e.kind in ("keep", "retract", "alias") or e.h in retracted:
                continue
            for target_h, w in e.displaces:
                if target_h in revoked or target_h in retracted or target_h not in row_of:
                    continue
                ops.append(("erase", row_of[target_h], float(w)))
            ops.append(("write", i, 0.0))
        return ops

    def _fold_ops(self, st: core.FoldState, ops: list[tuple],
                  K: np.ndarray, V: np.ndarray, chunk: int) -> core.FoldState:
        beta = self.cfg["beta"]
        run: list[int] = []

        def flush(state: core.FoldState) -> core.FoldState:
            if not run:
                return state
            out = core.fold(state, K[run].astype(np.float32),
                            V[run].astype(np.float32), beta, chunk)
            run.clear()
            return out

        for op, idx, w in ops:
            if op == "write":
                run.append(idx)
            else:
                st = flush(st)
                k = K[idx].astype(np.float32)
                st.S -= w * np.outer(st.S @ k, k)   # invertible attenuation (w < 1)
        return flush(st)

    def catch_up(self, embedder: Embedder | None = None, chunk: int = core.CHUNK,
                 quiet: bool = False, embed_missing: bool = True,
                 entries: list[Entry] | None = None) -> core.FoldState:
        """Embed and fold whatever the log has that derived data hasn't.
        The universal entry point: every command calls this first. A new
        `keep` event forces a clean refold so revocation is exact.
        embed_missing=False folds only rows whose vectors already exist —
        model-free, used for read-only peer mirrors.
        entries lets a caller that already parsed the log hand it in, so one
        command invocation doesn't re-read log.jsonl per step."""
        entries = (self._embed_missing(embedder, quiet, entries=entries) if embed_missing
                   else (entries if entries is not None else self.entries()))
        st = self.load_state()
        n_vec = min(self._rows_on_disk(VEC_V), self._rows_on_disk(VEC_K))
        if any(e.kind in ("keep", "retract") for e in entries[st.n:n_vec]):
            st = core.FoldState.empty(self.dim)
        if st.n < n_vec:
            K = self.vectors(VEC_K)
            V = self.vectors(VEC_V)
            ops = self._build_ops(entries[:n_vec], start=st.n)
            st = self._fold_ops(st, ops, K, V, chunk)
            st.n = n_vec
            self.save_state(st)
        vw = getattr(self, "views", None)
        if vw is None:
            vw = self.load_views()
        if vw.n > n_vec:
            vw = views.Views.empty()
        if vw.n < n_vec:
            vw = views.advance(vw, entries[:n_vec], vw.n)
            self.save_views(vw)
        self.views = vw
        return st

    # -------------------------------------------------------------- inference

    def remember_inferred(self, entry: Entry, embedder: Embedder | None = None,
                          ) -> tuple[bool, list[tuple[Entry, float]]]:
        """Append one entry, inferring which prior beliefs it displaces
        (keyless entries only — declared topics already have exact
        supersession). Returns (added, [(displaced_entry, weight), ...])."""
        if embedder is None:
            embedder = Embedder(self.cfg["model"])
        entries = self.entries()
        self.catch_up(embedder, quiet=True, entries=entries)
        if entry.h in {e.h for e in entries}:
            return False, []

        if entry.topic:
            # topic collision check: a legitimate version is semantically near
            # what it replaces; low similarity means a SIBLING is about to be
            # erased by accident (topic used as a folder, not a slot).
            last = self.latest_by_topic(entries)
            if entry.topic in last:
                prior_i = last[entry.topic]
                value_lift0, _ = self._lift_ops()
                v_new = value_lift0(embedder.passages([entry.text]))[0]
                v_old = self.vectors(VEC_V)[prior_i].astype(np.float32)
                cos = float(v_old @ v_new)
                prior = entries[prior_i]
                if cos < 0.55:
                    print(f"  ! topic '{entry.topic}' held UNRELATED content "
                          f"(cos {cos:.2f}) — this write erases {prior.h[:8]} "
                          f"\"{' '.join(prior.text.split())[:60]}\". If these should "
                          f"coexist, use sibling topics ({entry.topic}/a, "
                          f"{entry.topic}/b); undo with: mnema forget <this write>",
                          flush=True)
                else:
                    print(f"  supersedes {prior.h[:8]} at {entry.topic} "
                          f"(cos {cos:.2f})", flush=True)

        displaced: list[tuple[Entry, float]] = []
        if not entry.topic and entries:
            value_lift, _ = self._lift_ops()
            q = value_lift(embedder.passages([entry.text]))[0]   # keyless: k = v
            K = self.vectors(VEC_K).astype(np.float32)
            cos = K[: len(entries)] @ q
            order = np.argsort(-cos)[:INFER_TOPK]
            for j in order:
                c = float(cos[j])
                if c < INFER_FLOOR or entries[j].kind == "keep":
                    continue
                w = min(INFER_CAP, c)
                entry.displaces.append([entries[j].h, round(w, 3)])
                displaced.append((entries[j], w))

        self.append([entry])
        self.catch_up(embedder, quiet=True)
        return True, displaced

    def ingest_inferred(self, entries: list[Entry], embedder: Embedder | None = None,
                        ) -> tuple[int, int]:
        """Bulk append WITH displacement inference, computed in log order —
        the state ends identical to remembering each entry one at a time.
        Returns (entries_added, displacements_inferred)."""
        if embedder is None:
            embedder = Embedder(self.cfg["model"])
        existing = self.entries()
        self.catch_up(embedder, quiet=True, entries=existing)
        known = {e.h for e in existing}
        uniq: list[Entry] = []
        for e in entries:
            if e.h in known:
                continue
            known.add(e.h)
            uniq.append(e)
        if not uniq:
            return 0, 0

        value_lift, key_lift = self._lift_ops()
        V_new = value_lift(embedder.passages([e.text for e in uniq]))
        K_new = key_lift(embedder.passages([e.topic or e.text for e in uniq]))

        n0 = self._rows_on_disk(VEC_K)
        Kbuf = np.empty((n0 + len(uniq), self.dim), np.float32)
        Kbuf[:n0] = self.vectors(VEC_K).astype(np.float32)
        for i, e in enumerate(uniq):
            Kbuf[n0 + i] = K_new[i] = core.bind_slots(K_new[i], e.slots, self.cfg["seed"])

        # Displacement inference compares each new key against every
        # strictly-prior key — same candidate pool a sequential `remember`
        # would see. That's inherent (any prior belief can be displaced), but
        # doing it as one small matvec + argsort per entry is scalar-loop
        # overhead: O(len(uniq)) tiny BLAS calls on top of the O(N^2 D) work.
        # Block the new entries and run one wide GEMM per block instead — the
        # per-row top-k selection below stays a loop (cheap: INFER_TOPK=3),
        # only the FLOP-heavy comparison is batched.
        all_entries = existing + uniq
        n_disp = 0
        for b0 in range(0, len(uniq), INFER_BLOCK):
            b1 = min(b0 + INFER_BLOCK, len(uniq))
            cos_block = Kbuf[n0 + b0:n0 + b1] @ Kbuf[:n0 + b1].T
            for r in range(b1 - b0):
                i = b0 + r
                e = uniq[i]
                valid = n0 + i                      # causal cutoff: strictly-prior keys only
                if e.topic or valid == 0:
                    continue
                cos = cos_block[r, :valid]
                # top-INFER_TOPK by score, descending — argpartition is O(valid)
                # instead of argsort's O(valid log valid); summed over a bulk
                # load that's the difference between O(N^2) and O(N^2 log N).
                if valid > INFER_TOPK:
                    part = np.argpartition(-cos, INFER_TOPK - 1)[:INFER_TOPK]
                    top = part[np.argsort(-cos[part])]
                else:
                    top = np.argsort(-cos)
                for j in top:
                    c = float(cos[j])
                    target = all_entries[j]
                    if c < INFER_FLOOR or target.kind == "keep":
                        continue
                    e.displaces.append([target.h, round(min(INFER_CAP, c), 3)])
                    n_disp += 1

        added = self.append(uniq)
        self._append_vectors(VEC_V, V_new)
        self._append_vectors(VEC_K, K_new)
        self.catch_up(embedder, quiet=True)
        return added, n_disp

    def keep(self, hash_prefix: str, embedder: Embedder | None = None) -> Entry | None:
        """Revoke every inferred displacement of the entry matching the hash
        prefix, restoring it to full strength (exact: forces a refold)."""
        entries = self.entries()
        targets = {t for e in entries for t, _ in e.displaces}
        matches = [t for t in targets if t.startswith(hash_prefix)]
        if len(matches) != 1:
            return None
        by_h = {e.h: e for e in entries}
        self.append([Entry(text=f"keep {matches[0]}", kind="keep", target=matches[0])])
        self.catch_up(embedder, quiet=True)
        return by_h[matches[0]]

    # -------------------------------------------------------------- views

    def latest_by_topic(self, entries: list[Entry] | None = None) -> dict[str, int]:
        entries = entries if entries is not None else self.entries()
        retracted = self.retracted_hashes(entries)
        last: dict[str, int] = {}
        for i, e in enumerate(entries):
            if e.kind in ("keep", "retract", "alias") or e.h in retracted:
                continue
            last[e.topic or f"__anon__{i}"] = i
        return last

    def forget(self, targets: list[str], entries: list[Entry] | None = None) -> list[Entry]:
        """Retract entries by hash: append retraction events and refold.
        The state returns to exactly what it would be had they never been
        written — reversal by re-derivation from the corrected event set."""
        entries = entries if entries is not None else self.entries()
        already = self.retracted_hashes(entries)
        by_h = {e.h: e for e in entries if e.kind not in ("keep", "retract")}
        resolved: list[Entry] = []
        for prefix in targets:
            matches = [h for h in by_h if h.startswith(prefix) and h not in already]
            if len(matches) != 1:
                raise SystemExit(f"hash prefix '{prefix}' matches "
                                 f"{len(matches)} entries — need exactly 1")
            resolved.append(by_h[matches[0]])
        self.append([Entry(text=f"retract {e.h}", kind="retract", target=e.h)
                     for e in resolved])
        self.catch_up(quiet=True)                 # model-free: forces exact refold
        return resolved

    def fold_prefix(self, upto_at: str, entries: list[Entry] | None = None) -> core.FoldState:
        """Materialize the state as of a timestamp (time travel). Refolds the
        prefix with the chunked rule — O(n D^2) as GEMMs, seconds in practice."""
        entries = entries if entries is not None else self.entries()
        idx = [i for i, e in enumerate(entries) if e.at <= upto_at]
        st = core.FoldState.empty(self.dim)
        if not idx:
            return st
        K = self.vectors(VEC_K)[idx].astype(np.float32)
        V = self.vectors(VEC_V)[idx].astype(np.float32)
        return core.fold(st, K, V, self.cfg["beta"])


def merge(a: Store, b: Store, out_path: Path,
         a_entries: list[Entry] | None = None,
         b_entries: list[Entry] | None = None) -> Store:
    """Merge two stores with identical configs into a new store: interleave
    logs by timestamp, dedupe, reuse both stores' vectors (no re-embedding),
    fold once. Associativity is what makes this lawful."""
    for key in ("model", "dim", "seed", "beta", "sigma", "d_embed"):
        if a.cfg[key] != b.cfg[key]:
            raise SystemExit(f"config mismatch on '{key}': {a.cfg[key]} != {b.cfg[key]}")
    rows: dict[str, tuple[Entry, Store, int]] = {}
    for s, s_entries in ((a, a_entries), (b, b_entries)):
        for i, e in enumerate(s_entries if s_entries is not None else s.entries()):
            rows.setdefault(e.h, (e, s, i))
    ordered = sorted(rows.values(), key=lambda t: (t[0].at, t[0].h))

    out = Store.init(out_path, a.cfg["model"], a.cfg["dim"], a.cfg["beta"],
                     a.cfg["sigma"], a.cfg["seed"], a.cfg["d_embed"])
    out.append([e for e, _, _ in ordered])
    for name, sel in ((VEC_V, 0), (VEC_K, 1)):
        # one memmap per source store, not one per row: `vectors()` opens and
        # reshapes the file each call, and ordered can hold thousands of rows.
        vecs_by_store = {s: s.vectors(name) for s in (a, b)}
        chunks = [np.asarray(vecs_by_store[src][i]) for _, src, i in ordered]
        out._append_vectors(name, np.stack(chunks))
    out.catch_up(quiet=True)
    return out
