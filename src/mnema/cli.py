"""mnema — constant-size semantic memory over any append-only knowledge log.

    mnema init   [--store PATH] [--model NAME] [--dim 4096] [--sigma S]
    mnema remember "text" [--topic KEY] [--slot name=value ...]
    mnema ingest --format ledger|jsonl PATH [PATH ...]
    mnema ask "question" [--top 5] [--as-of ISO] [--current] [--slot n=v ...]
    mnema stats
    mnema merge OTHER_STORE --out NEW_STORE
"""

from __future__ import annotations

import argparse
import glob as globlib
import os
import textwrap
from pathlib import Path

from . import core
from .embed import DEFAULT_MODEL, Embedder, MODELS
from .store import Entry, Store, merge, past_horizon
from .translate import TRANSLATORS

DEFAULT_STORE = Path(os.environ.get("MNEMA_STORE", Path.home() / ".mnema"))


def _slots(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs or []:
        name, _, value = p.partition("=")
        if not value:
            raise SystemExit(f"--slot must be name=value, got: {p}")
        out[name] = value
    return out


def cmd_init(args) -> None:
    embedder = Embedder(args.model)
    Store.init(Path(args.store), args.model, args.dim, args.beta, args.sigma,
               args.seed, embedder.dim)
    print(f"initialized {args.store} (model={args.model}, D={args.dim}, "
          f"beta={args.beta}, sigma={args.sigma})")


def cmd_remember(args) -> None:
    store = Store(Path(args.store))
    e = Entry(text=args.text, topic=args.topic, slots=_slots(args.slot),
              questions=args.question or [])
    added, displaced = store.remember_inferred(e)
    label = f" @ {args.topic}" if args.topic else ""
    print(f"{'remembered' if added else 'already known'}{label} ({e.at})")
    if added and past_horizon(e.text):
        print("  ! long memory: content past ~350 words never reaches the index "
              "— split into separate beliefs for full retrievability")
    for old, w in displaced:
        snippet = " ".join(old.text.split())[:80]
        print(f"  displaced {old.h[:8]}  [attenuated {w:.2f}]  \"{snippet}\"")
    if displaced:
        print(f"  (wrong? restore with: mnema keep <hash>)")


def cmd_keep(args) -> None:
    store = Store(Path(args.store))
    restored = store.keep(args.hash)
    if restored is None:
        raise SystemExit(f"no unique displaced entry matches hash prefix '{args.hash}'")
    snippet = " ".join(restored.text.split())[:80]
    print(f"restored {restored.h[:8]} to full strength: \"{snippet}\"")


def cmd_ingest(args) -> None:
    import sys
    from .translate import generic_jsonl_lines
    store = Store(Path(args.store))
    translate = TRANSLATORS[args.format]
    entries = []
    for pattern in args.paths:
        if pattern == "-":
            if args.format != "jsonl":
                raise SystemExit("stdin ingest is jsonl only — transform with "
                                 "your pipeline first (jq/awk/anything), then pipe")
            entries.extend(generic_jsonl_lines(sys.stdin))
            continue
        for p in sorted(globlib.glob(pattern, recursive=True)):
            entries.extend(translate(Path(p)))
    entries.sort(key=lambda e: (e.at, e.h))
    long_n = sum(1 for e in entries if past_horizon(e.text))
    if args.infer:
        added, n_disp = store.ingest_inferred(entries)
        print(f"{len(entries)} entries from {args.format} source(s), {added} new, "
              f"{n_disp} displacements inferred")
    else:
        added = store.append(entries)
        print(f"{len(entries)} entries from {args.format} source(s), {added} new")
        store.catch_up()
    if long_n:
        print(f"  ! {long_n} entries exceed the ~350-word index horizon — their "
              f"tails are not retrievable; consider a finer-grained translator")
    print("folded.")


def cmd_ask(args) -> None:
    from .query import ask
    store = Store(Path(args.store))
    scope = "local" if args.local else (args.from_vault or "all")
    ans = ask(store, args.question, top=args.top, as_of=args.as_of,
              current_only=args.current, slots=_slots(args.slot), vaults=scope)
    when = f" as of {args.as_of}" if args.as_of else ""
    VERDICT_LINE = {
        "settled": "settled — a strong match exists; the top result is trustworthy",
        "sparse": "sparse — likely in the results below; read critically",
        "unwritten": "unwritten — nothing matches this well; nearest content "
                     "shown, verify before trusting",
    }
    if args.scores:
        from .query import SUPPORT_SETTLED, SUPPORT_UNWRITTEN
        band = {"settled": f"≥{SUPPORT_SETTLED}", "unwritten": f"<{SUPPORT_UNWRITTEN}",
                "sparse": f"{SUPPORT_UNWRITTEN}–{SUPPORT_SETTLED}"}[ans.verdict]
        print(f"support {ans.support} (band {band}){when}")
    print(f"{VERDICT_LINE[ans.verdict]}{when if not args.scores else ''}")
    for w in ans.warnings:
        print(f"  ! {w}")
    for h in ans.hits:
        tag = f"  <<superseded by {h.superseded_by[:10]} write>>" if h.superseded_by else ""
        if h.displaced:
            tag += f"  <<displaced {h.displaced:.2f} by later write>>"
        origin = "" if h.source == "local" else f"  (vault: {h.source})"
        score = f"{h.score:6.3f}  " if args.scores else ""
        head = f"\n  {score}{h.at[:10]}  [{h.kind}] {h.topic or '—'} {h.h[:8]}{origin}{tag}"
        print(head)
        print(textwrap.fill(" ".join(h.text.split()), width=100,
                            initial_indent="          ", subsequent_indent="          "))


def cmd_log(args) -> None:
    from . import vaults as V
    store = Store(Path(args.store))
    source = "local"
    if args.from_vault:
        match = [v for v in V.load_vaults(store.path) if v["name"] == args.from_vault]
        if not match:
            raise SystemExit(f"no vault named '{args.from_vault}' (see: mnema vault list)")
        store = Store(V.sync_vault(store.path, match[0]))
        source = args.from_vault

    retracted = store.retracted_hashes()
    entries = [e for e in store.entries()
               if e.kind not in ("keep", "retract") and e.h not in retracted]
    last = store.latest_by_topic()
    revoked = {e.target for e in store.entries() if e.kind == "keep" and e.target}
    displaced = {}
    for e in store.entries():
        for target_h, w in e.displaces:
            if target_h not in revoked:
                displaced[target_h] = max(displaced.get(target_h, 0.0), w)

    order = sorted(range(len(entries)), key=lambda i: entries[i].at, reverse=True)
    shown = order if args.all else order[: args.top]
    print(f"{source}: {len(entries)} memories, newest first"
          + ("" if args.all else f" (top {len(shown)}; --all for everything)"))
    for i in shown:
        e = entries[i]
        tag = ""
        if e.topic and last.get(e.topic) is not None and entries[last[e.topic]].h != e.h:
            tag += "  <<superseded>>"
        if e.h in displaced:
            tag += f"  <<displaced {displaced[e.h]:.2f}>>"
        print(f"\n  {e.at[:19]}  [{e.kind}] {e.topic or '—'} {e.h[:8]}{tag}")
        text = " ".join(e.text.split())
        if not args.full and len(text) > 160:
            text = text[:157] + "..."
        print(textwrap.fill(text, width=100, initial_indent="          ",
                            subsequent_indent="          "))


def cmd_forget(args) -> None:
    store = Store(Path(args.store))
    targets = list(args.hashes)
    if args.last:
        retracted = store.retracted_hashes()
        live = [e for e in store.entries()
                if e.kind not in ("keep", "retract") and e.h not in retracted]
        targets.extend(e.h for e in live[-args.last:])
    if not targets:
        raise SystemExit("nothing to forget: pass hash prefixes and/or --last N")
    forgotten = store.forget(targets)
    for e in forgotten:
        snippet = " ".join(e.text.split())[:80]
        print(f"forgot {e.h[:8]}  \"{snippet}\"")
    print(f"({len(forgotten)} retracted; state refolded as if never written)")


def cmd_serve(args) -> None:
    from .embed import Embedder
    from .serve import serve
    model = args.model
    if not model:
        cfg_path = Path(args.store) / "config.json"
        model = (Store(Path(args.store)).cfg["model"] if cfg_path.exists()
                 else DEFAULT_MODEL)
    serve(Embedder(model, direct=True))


def cmd_vault(args) -> None:
    from . import vaults as V
    store = Store(Path(args.store))
    if args.action == "add":
        v = V.add_vault(store.path, args.addr, args.name)
        print(f"vault '{v['name']}' added: {v['addr']}")
    elif args.action == "remove":
        ok = V.remove_vault(store.path, args.addr)
        print(f"vault '{args.addr}' {'removed' if ok else 'not found'}")
    else:  # list
        entries = store.entries()
        print(f"{'local':<16} {str(store.path):<50} {len(entries)} memories")
        for v in V.load_vaults(store.path):
            try:
                vs = Store(V.sync_vault(store.path, v))
                detail = f"{len(vs.entries())} memories"
            except Exception as ex:
                detail = f"unreachable ({type(ex).__name__})"
            print(f"{v['name']:<16} {v['addr']:<50} {detail}")


def cmd_stats(args) -> None:
    store = Store(Path(args.store))
    st = store.catch_up(quiet=True)
    entries = store.entries()
    topics = {e.topic for e in entries if e.topic}
    gauge = core.spectral_gauge(st)
    sizes = {f.name: f.stat().st_size for f in Path(args.store).iterdir() if f.is_file()}
    print(f"store        {args.store}")
    print(f"entries      {len(entries)}  (topics: {len(topics)}, "
          f"anonymous: {sum(1 for e in entries if not e.topic)})")
    print(f"model        {store.cfg['model']}  D={store.cfg['dim']}  "
          f"beta={store.cfg['beta']}  sigma={store.cfg['sigma']}")
    print(f"gauge        live-topics≈{gauge['live_topics_estimate']}  "
          f"headroom={gauge['headroom']:.0%}  "
          f"eff-rank(S)≈{gauge['effective_rank_sketch']}")
    print(f"top σ        {gauge['top_singular_values']}")
    total = sum(sizes.values())
    print(f"disk         {total / 1e6:.1f} MB  " +
          "  ".join(f"{k}={v / 1e6:.1f}MB" for k, v in sorted(sizes.items())))
    if gauge["headroom"] < 0.2:
        print("WARNING: <20% capacity headroom — raise D or shard this store")


def cmd_merge(args) -> None:
    a, b = Store(Path(args.store)), Store(Path(args.other))
    a.catch_up(quiet=True)
    b.catch_up(quiet=True)
    merged = merge(a, b, Path(args.out))
    print(f"merged {args.store} + {args.other} -> {args.out} "
          f"({len(merged.entries())} entries)")


def main() -> None:
    ap = argparse.ArgumentParser(prog="mnema", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=str(DEFAULT_STORE),
                    help=f"store directory (default {DEFAULT_STORE}, or $MNEMA_STORE)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init", help="create a store")
    p.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODELS) + ["custom"],
                   help="embedding model")
    p.add_argument("--dim", type=int, default=4096, help="fold dimension D (capacity ≈ D live topics)")
    p.add_argument("--beta", type=float, default=1.0, help="1.0 = exact overwrite; <1 = soft supersession")
    p.add_argument("--sigma", type=float, default=None, help="RFF bandwidth for key locality (default: linear lift)")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("remember", help="append one knowledge entry (O(1)); keyless "
                       "entries infer + attenuate the prior beliefs they displace")
    p.add_argument("text")
    p.add_argument("--topic", help="address; a later write to the same topic supersedes this one")
    p.add_argument("--slot", action="append", help="name=value structure bound into the key")
    p.add_argument("--question", action="append",
                   help="optional: a question this memory answers (extra address)")
    p.set_defaults(fn=cmd_remember)

    p = sub.add_parser("keep", help="revoke an inferred displacement; restores the entry")
    p.add_argument("hash", help="hash prefix printed by remember/ask")
    p.set_defaults(fn=cmd_keep)

    p = sub.add_parser("forget", help="retract entries: exact undo, as if never written")
    p.add_argument("hashes", nargs="*", help="hash prefixes to retract")
    p.add_argument("--last", type=int, help="also retract the N most recent entries")
    p.set_defaults(fn=cmd_forget)

    p = sub.add_parser("ingest", help="bulk-append entries via a translator")
    p.add_argument("--format", choices=sorted(TRANSLATORS), required=True)
    p.add_argument("--infer", action="store_true",
                   help="run displacement inference per entry, in order — the "
                        "state ends identical to remembering one at a time")
    p.add_argument("paths", nargs="+", help="files or globs")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("ask", help="query the folded state (local + vaults)")
    p.add_argument("question")
    p.add_argument("--top", type=int, default=5)
    p.add_argument("--as-of", dest="as_of", help="ISO timestamp: answer from the state as of then")
    p.add_argument("--current", action="store_true", help="hard-filter to latest write per topic")
    p.add_argument("--slot", action="append", help="name=value to bind into the query address")
    p.add_argument("--from", dest="from_vault", metavar="VAULT",
                   help="ask one named vault only (e.g. --from alice)")
    p.add_argument("--local", action="store_true", help="skip vaults entirely")
    p.add_argument("--scores", action="store_true",
                   help="show raw support and per-hit cosines (diagnostics)")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("serve", help="warm-model daemon: makes ask/remember ~instant")
    p.add_argument("--model", help="model to keep warm (default: the store's model)")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("log", help="list memories, newest first (local or a vault)")
    p.add_argument("--from", dest="from_vault", metavar="VAULT",
                   help="list a named vault instead of the local store")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--all", action="store_true", help="no limit")
    p.add_argument("--full", action="store_true", help="full text, no truncation")
    p.set_defaults(fn=cmd_log)

    p = sub.add_parser("vault", help="named read-only memories that join your queries")
    p.add_argument("action", choices=["add", "list", "remove"])
    p.add_argument("addr", nargs="?", help="add: URL or path; remove: vault name")
    p.add_argument("--name", help="vault name (default: derived from address)")
    p.set_defaults(fn=cmd_vault)

    p = sub.add_parser("stats", help="entries, capacity gauge, disk")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("merge", help="merge two same-config stores into a new one")
    p.add_argument("other")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_merge)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
