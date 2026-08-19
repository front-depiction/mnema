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
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import core, views
from .embed import Embedder

CONFIG, LOG, VEC_V, VEC_K, STATE = "config.json", "log.jsonl", "vec_v.f16", "vec_k.f16", "state.npz"
VIEWS = "views.npz"
LOG_OFF = "log.off"                    # int64 (start, end) byte span per line
PRE_K, PRE_V = "pre_k.f16", "pre_v.f16"

# Prefilter sidecar: a seeded projection of the lifted rows down to PRE_DIM,
# appended in lockstep with the vector files. At library scale the query's
# coarse scan reads this (PRE_DIM/dim of the bytes) and only nominated rows
# are touched at full width — the rescore is exact, so shown scores are true
# cosines and only the candidate cut is approximate.
PRE_DIM = 256

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

# Bulk embedding runs in blocks: one fused forward pass per block computes
# values AND keys (keyless entries share the text — embed once, lift twice),
# then both vector files checkpoint before the next block. A killed ingest
# resumes from the last block; memory stays flat regardless of corpus size.
EMBED_BLOCK = 512

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

    def line_offsets(self) -> np.ndarray:
        """(start, end) byte span of every non-blank log line, as an (n, 2)
        int64 array. Cached in an append-only sidecar; the log is append-only,
        so advancing scans only the unseen tail — O(delta) per command."""
        p = self.path / LOG
        size = p.stat().st_size if p.exists() else 0
        op = self.path / LOG_OFF
        spans = (np.fromfile(op, np.int64).reshape(-1, 2) if op.exists()
                 else np.empty((0, 2), np.int64))
        done = int(spans[-1, 1]) if len(spans) else 0
        if done > size:                                    # log replaced: rebuild
            spans, done = np.empty((0, 2), np.int64), 0
        if done < size:
            fresh = []
            with p.open("rb") as f:                    # stream: no whole-file read
                f.seek(done)
                carry = b""
                base = done
                while True:
                    buf = f.read(64 * 1024 * 1024)
                    if not buf:
                        if carry.strip():              # final line without newline
                            fresh.append((base, base + len(carry)))
                        break
                    buf = carry + buf
                    pos = 0
                    while True:
                        nl = buf.find(b"\n", pos)
                        if nl < 0:
                            break
                        if buf[pos:nl].strip():
                            fresh.append((base + pos, base + nl))
                        pos = nl + 1
                    carry = buf[pos:]
                    base += pos
            if fresh:
                new = np.asarray(fresh, np.int64)
                spans = np.concatenate([spans, new]) if len(spans) else new
                try:
                    with op.open("ab" if done else "wb") as f:
                        (new if done else spans).tofile(f)
                except OSError:
                    pass                                   # read-only mirror
        return spans

    def entry_count(self) -> int:
        return len(self.line_offsets())

    def read_rows(self, rows: list[int]) -> list[Entry]:
        """Parse exactly these log lines, located by the offset sidecar —
        an ask fetches the handful of rows it displays, not the corpus."""
        spans = self.line_offsets()
        out = []
        with (self.path / LOG).open("rb") as f:
            for i in rows:
                start, end = int(spans[i, 0]), int(spans[i, 1])
                f.seek(start)
                out.append(Entry.from_line(f.read(end - start).decode()))
        return out

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

    def _rows_on_disk(self, name: str, width: int | None = None) -> int:
        p = self.path / name
        return p.stat().st_size // (2 * (width or self.dim)) if p.exists() else 0

    def vectors(self, name: str) -> np.ndarray:
        p = self.path / name
        if not p.exists() or p.stat().st_size == 0:
            return np.empty((0, self.dim), np.float16)
        return np.memmap(p, dtype=np.float16, mode="r").reshape(-1, self.dim)

    def _append_vectors(self, name: str, rows: np.ndarray) -> None:
        with (self.path / name).open("ab") as f:
            f.write(rows.astype(np.float16).tobytes())

    def _pre_projection(self) -> np.ndarray:
        return core.lift_projection(self.dim, PRE_DIM, self.cfg["seed"] + 2)

    def prefilter(self, n: int):
        """(pre_k, pre_v, P) when the sidecar covers n rows, else None — the
        caller falls back to the exact scan, so a lagging sidecar only costs
        speed, never correctness."""
        if (self._rows_on_disk(PRE_K, PRE_DIM) < n
                or self._rows_on_disk(PRE_V, PRE_DIM) < n):
            return None
        pk = np.memmap(self.path / PRE_K, np.float16, "r").reshape(-1, PRE_DIM)
        pv = np.memmap(self.path / PRE_V, np.float16, "r").reshape(-1, PRE_DIM)
        return pk, pv, self._pre_projection()

    def _advance_prefilter(self, block: int = 8192) -> None:
        """Project any vector rows the sidecar hasn't seen — pure derivation
        from the vec files, no model, O(delta)."""
        P = None
        for src, dst in ((VEC_K, PRE_K), (VEC_V, PRE_V)):
            n_src = self._rows_on_disk(src)
            n_dst = self._rows_on_disk(dst, PRE_DIM)
            if n_dst > n_src:                           # replaced vectors: rebuild
                (self.path / dst).unlink(missing_ok=True)
                n_dst = 0
            if n_dst >= n_src:
                continue
            if P is None:
                P = self._pre_projection()
            mm = self.vectors(src)
            with (self.path / dst).open("ab") as f:
                for a in range(n_dst, n_src, block):
                    X = np.asarray(mm[a:a + block], np.float32) @ P
                    norms = np.linalg.norm(X, axis=1, keepdims=True)
                    norms[norms == 0] = 1.0             # bookkeeping rows stay zero
                    f.write((X / norms).astype(np.float16).tobytes())

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

    def _repair_torn_append(self) -> int:
        """Vector rows align positionally with log lines; a crash between the
        V and K appends of a block leaves one file longer. Truncate to the
        aligned prefix so the next append can't shift every later row."""
        n = min(self._rows_on_disk(VEC_V), self._rows_on_disk(VEC_K))
        for name in (VEC_V, VEC_K):
            p = self.path / name
            if p.exists() and p.stat().st_size > n * 2 * self.dim:
                os.truncate(p, n * 2 * self.dim)
        return n

    def _embed_missing(self, embedder: Embedder | None, quiet: bool,
                       entries: list[Entry] | None = None,
                       block: int | None = None) -> list[Entry]:
        entries = entries if entries is not None else self.entries()
        n_vec = self._repair_torn_append()
        missing = entries[n_vec:]
        if not missing:
            return entries
        if embedder is None:
            embedder = Embedder(self.cfg["model"])
        if not quiet:
            print(f"embedding {len(missing)} new entries...", flush=True)
        value_lift, key_lift = self._lift_ops()
        row_of = {e.h: i for i, e in enumerate(entries)}
        block = block or EMBED_BLOCK
        for b0 in range(0, len(missing), block):
            blk = missing[b0:b0 + block]
            # one fused pass per block: every unique text crosses the model
            # once, then both lifts read from the shared embedding matrix
            v_text: dict[int, str] = {}
            k_text: dict[int, str] = {}
            for i, e in enumerate(blk):
                if e.kind in ("keep", "retract"):
                    continue                  # bookkeeping events hold zero rows
                k_text[i] = e.topic or e.text
                if e.kind != "alias":
                    v_text[i] = e.text        # alias values copy from the parent
            uniq = list(dict.fromkeys(list(v_text.values()) + list(k_text.values())))
            V = np.zeros((len(blk), self.dim), np.float32)
            K = np.zeros((len(blk), self.dim), np.float32)
            if uniq:
                E = embedder.passages(uniq)
                pos = {t: j for j, t in enumerate(uniq)}
                EV, EK = value_lift(E), key_lift(E)
                for i, e in enumerate(blk):
                    if i in k_text:
                        K[i] = core.bind_slots(EK[pos[k_text[i]]], e.slots,
                                               self.cfg["seed"])
                    if i in v_text:
                        V[i] = EV[pos[v_text[i]]]
            base = n_vec + b0
            Vd = self.vectors(VEC_V)          # all prior blocks are on disk
            for i, e in enumerate(blk):
                if e.kind == "alias" and e.target:
                    r = row_of.get(e.target)
                    if r is None:
                        continue
                    if r < base:
                        V[i] = Vd[r].astype(np.float32)
                    elif r - base < len(blk):
                        V[i] = V[r - base]    # parent precedes its alias in the log
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
        try:
            self._advance_prefilter()
        except OSError:
            pass                       # read-only mirror: exact scan still serves
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
                          f"(cos {cos:.2f}) — this write erases it entirely:",
                          flush=True)
                    print(f'<erased h="{prior.h[:8]}" at="{prior.at[:10]}" '
                          f'topic="{prior.topic}">', flush=True)
                    print("  " + " ".join(prior.text.split()), flush=True)
                    print("</erased>", flush=True)
                    print(f"  if these should coexist, use sibling topics "
                          f"({entry.topic}/a, {entry.topic}/b); undo this write "
                          f"with: mnema forget <its hash>", flush=True)
                else:
                    print(f"  supersedes {prior.h[:8]} at {entry.topic} "
                          f"(cos {cos:.2f})", flush=True)

        displaced: list[tuple[Entry, float]] = []
        if not entry.topic and entries:
            value_lift, _ = self._lift_ops()
            q = value_lift(embedder.passages([entry.text]))[0]   # keyless: k = v
            K = self.vectors(VEC_K)
            cos = np.empty(min(len(entries), K.shape[0]), np.float32)
            for a in range(0, len(cos), 65536):                  # blocked: flat memory
                cos[a:a + 65536] = np.asarray(K[a:a + 65536], np.float32) @ q
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
        v_texts = [e.text for e in uniq]
        k_texts = [e.topic or e.text for e in uniq]
        # one fused pass: keyless entries share value and key text, so the
        # union is embedded once and both lifts read the shared matrix
        E = embedder.passages(list(dict.fromkeys(v_texts + k_texts)))
        pos = {t: j for j, t in enumerate(dict.fromkeys(v_texts + k_texts))}
        V_new = value_lift(E[[pos[t] for t in v_texts]])
        K_new = key_lift(E[[pos[t] for t in k_texts]])

        n0 = self._rows_on_disk(VEC_K)
        Kd = self.vectors(VEC_K)               # prior keys stay on disk, fp16
        for i, e in enumerate(uniq):
            K_new[i] = core.bind_slots(K_new[i], e.slots, self.cfg["seed"])

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
            Kb = K_new[b0:b1]
            # prior keys stream from disk block-by-block; new keys are in RAM
            cos_block = np.empty((b1 - b0, n0 + b1), np.float32)
            for a in range(0, n0, 65536):
                b = min(a + 65536, n0)
                cos_block[:, a:b] = Kb @ np.asarray(Kd[a:b], np.float32).T
            cos_block[:, n0:n0 + b1] = Kb @ K_new[:b1].T
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

    def add_questions(self, hash_prefix: str, questions: list[str],
                      embedder: Embedder | None = None) -> tuple[Entry | None, int]:
        """Attach question-shaped addresses to an EXISTING memory: one alias
        entry per new question, targeting it. Idempotent per (target, text).
        Returns (target entry, aliases added); (None, 0) if no unique match."""
        entries = self.entries()
        retracted = self.retracted_hashes(entries)
        live = [e for e in entries if e.kind not in ("keep", "retract", "alias")
                and e.h not in retracted]
        matches = [e for e in live if e.h.startswith(hash_prefix)]
        if len(matches) != 1:
            return None, 0
        target = matches[0]
        have = {e.text for e in entries if e.kind == "alias" and e.target == target.h}
        fresh = [q for q in dict.fromkeys(questions) if q not in have]
        if not fresh:
            return target, 0
        self.append([Entry(text=q, kind="alias", target=target.h) for q in fresh])
        self.catch_up(embedder, quiet=True)
        return target, len(fresh)

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
        idx = np.asarray([i for i, e in enumerate(entries) if e.at <= upto_at])
        st = core.FoldState.empty(self.dim)
        if not len(idx):
            return st
        K, V = self.vectors(VEC_K), self.vectors(VEC_V)
        for a in range(0, len(idx), 8192):     # sequential fold: blocks compose
            sel = idx[a:a + 8192]
            st = core.fold(st, np.asarray(K[sel], np.float32),
                           np.asarray(V[sel], np.float32), self.cfg["beta"])
        return st


def recompile(store: Store, out_path: Path, model: str | None = None,
              embedder: Embedder | None = None) -> Store:
    """Rebuild a store's derived data from its log alone — optionally onto a
    different embedding model. The log is the authority and vectors/state are
    disposable caches, so a model migration is just: same log, fresh caches.
    Fold geometry (dim, beta, sigma, seed) carries over; d_embed follows the
    model. Blocked embedding makes an interrupted recompile resumable: rerun
    with the same out_path and it continues from the last checkpoint."""
    model = model or store.cfg["model"]
    if embedder is None:
        embedder = Embedder(model)
    if (out_path / CONFIG).exists():
        out = Store(out_path)                 # resume an interrupted recompile
        want = {"model": model, "dim": store.cfg["dim"], "beta": store.cfg["beta"],
                "sigma": store.cfg["sigma"], "seed": store.cfg["seed"]}
        stale = [k for k, v in want.items() if out.cfg.get(k) != v]
        if stale:
            raise SystemExit(f"{out_path} exists with different config ({stale}) — "
                             f"remove it or pick another --out")
    else:
        out = Store.init(out_path, model, store.cfg["dim"], store.cfg["beta"],
                         store.cfg["sigma"], store.cfg["seed"], embedder.dim)
        (out.path / LOG).write_text((store.path / LOG).read_text())
        for extra in ("vaults.json",):
            src = store.path / extra
            if src.exists():
                (out.path / extra).write_text(src.read_text())
    out.catch_up(embedder, quiet=True)
    return out


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
        # one memmap per source store, and rows flushed in blocks: merging
        # millions of rows stays memory-flat.
        vecs_by_store = {s: s.vectors(name) for s in (a, b)}
        for a0 in range(0, len(ordered), 8192):
            blk = ordered[a0:a0 + 8192]
            out._append_vectors(name, np.stack(
                [np.asarray(vecs_by_store[src][i]) for _, src, i in blk]))
    out.catch_up(quiet=True)
    return out
