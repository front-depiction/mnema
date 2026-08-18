"""mnema — constant-size semantic memory over any append-only knowledge log.

    mnema init   [--store PATH] [--model NAME] [--dim 4096] [--sigma S]
    mnema remember "text" [--topic KEY] [--slot name=value ...]
    mnema ingest --format ledger|jsonl PATH [PATH ...]
    mnema ask "question" [--top 5] [--as-of ISO] [--current] [--slot n=v ...]
    mnema show HASH [HASH ...]
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
        topic = f' topic="{old.topic}"' if old.topic else ""
        print(f'\n<displaced h="{old.h[:8]}" at="{old.at[:10]}"{topic} '
              f'attenuated="{w:.2f}" reading="{_attenuation_reading(w)}">')
        print(textwrap.fill(" ".join(old.text.split()), width=100,
                            initial_indent="  ", subsequent_indent="  "))
        print("</displaced>")
    if displaced:
        print("\nread each displaced memory: does the new one actually REPLACE it? "
              "if not, restore it: mnema keep <hash>")


def _attenuation_reading(w: float) -> str:
    """The weight in words: how directly the new memory contradicts the old
    (cosine at write, floor INFER_FLOOR, cap INFER_CAP)."""
    if w >= 0.88:
        return "near-restatement — almost certainly the same belief, updated"
    if w >= 0.80:
        return "strong overlap — likely a replacement, verify"
    return "borderline — same neighborhood, may be a sibling not a version"


def cmd_question(args) -> None:
    """Attach questions an existing memory answers — extra addresses for it."""
    store = Store(Path(args.store))
    target, added = store.add_questions(args.hash, args.questions)
    if target is None:
        raise SystemExit(f"no unique live memory matches hash prefix '{args.hash}'")
    snippet = " ".join(target.text.split())[:80]
    print(f"{added} question{'s' if added != 1 else ''} attached to {target.h[:8]}"
          f"  \"{snippet}\"" + ("" if added else "  (all already attached)"))


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
    if args.at:
        from dataclasses import replace
        entries = [replace(e, at=args.at) for e in entries]
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
    excluded = tuple(getattr(args, "except_vaults", None) or ())
    if excluded and args.from_vault:
        raise SystemExit("--from and --except together: pick one")
    ans = ask(store, args.question, top=args.top, as_of=args.as_of,
              current_only=args.current, slots=_slots(args.slot), vaults=scope,
              exclude=excluded)
    when = f" as of {args.as_of}" if args.as_of else ""
    VERDICT_LINE = {
        "settled": "settled — a strong match exists; the top result is trustworthy",
        "sparse": "sparse — likely in the results below; read critically",
        "unwritten": "unwritten — nothing matches this well; nearest content "
                     "shown, verify before trusting",
    }
    out: list[str] = []
    if args.scores:
        from .query import SUPPORT_SETTLED, SUPPORT_UNWRITTEN
        band = {"settled": f"≥{SUPPORT_SETTLED}", "unwritten": f"<{SUPPORT_UNWRITTEN}",
                "sparse": f"{SUPPORT_UNWRITTEN}–{SUPPORT_SETTLED}"}[ans.verdict]
        out.append(f"support {ans.support} (band {band}){when}")
    out.append(f"{VERDICT_LINE[ans.verdict]}{when if not args.scores else ''}")
    count_at = len(out)                       # the line-count line goes here
    out.append("navigate: <related/> lines are your answer's connections across every "
               "vault — mnema show <hash> opens any memory in full · "
               "mnema ask --from <vault> \"…\" scopes one · --except <vault> drops one · "
               "refine the question to keep moving")
    for w in ans.warnings:
        out.append(f"! {w}")
    # XML-shaped for readability, deliberately not parseable XML: attributes
    # are metadata, bodies are raw prose — never escaped.
    for rank, h in enumerate(ans.hits):
        attrs = [f'h="{h.h[:8]}"', f'at="{h.at[:10]}"', f'kind="{h.kind}"']
        if h.source != "local":
            attrs.append(f'vault="{h.source}"')
        if h.topic:
            attrs.append(f'topic="{h.topic}"')
        if h.superseded_by:
            attrs.append(f'superseded="{h.superseded_by[:10]}"')
        if h.displaced:
            attrs.append(f'displaced="{h.displaced:.2f}"')
        if args.scores:
            attrs.append(f'score="{h.score:.3f}"')
        out.append("")
        out.append(f"<hit {' '.join(attrs)}>")
        out.append(textwrap.fill(" ".join(h.text.split()), width=100,
                                 initial_indent="  ", subsequent_indent="  "))
        # cos always shown: a relatedness weight, not a support score
        for r in (ans.related_by_hit[rank] if rank < len(ans.related_by_hit) else []):
            rattrs = [f'h="{r.h[:8]}"', f'vault="{r.source}"',
                      f'cos="{r.cos:.2f}"']
            if args.scores:
                rattrs.append(f'score="{r.score:.3f}"')
            rattrs.append(f'gloss="{_gloss(r)}"')
            out.append(f"  <related {' '.join(rattrs)}/>")
        out.append("</hit>")
    # Rendered to a buffer first so the true total can lead: an agent about to
    # truncate sees the whole size before it cuts.
    total = sum(line.count("\n") + 1 for line in out) + 1
    out.insert(count_at, f"output: {total} lines — read all of them: memories are "
                         f"small and the answer is often in a later hit; truncating "
                         f"cuts relevant ones")
    print("\n".join(out))


def _gloss(r) -> str:
    """The one-line handle on a relation: its topic verbatim (a doc's heading
    slug IS the gloss), else the opening words of the memory."""
    if r.topic:
        path, sep, slug = r.topic.rpartition("#")
        return f"{path.rsplit('/', 1)[-1]}#{slug}" if sep else r.topic
    words = r.text.split()
    return " ".join(words[:15]) + ("…" if len(words) > 15 else "")


def cmd_show(args) -> None:
    from . import vaults as V
    store = Store(Path(args.store))

    sources: list[tuple[str, list[Entry]]] = [("local", store.entries())]
    for vault in V.load_vaults(store.path):
        try:
            mirror = V.sync_vault(store.path, vault)
        except Exception as ex:
            mirror = Path(store.path) / "vaults" / vault["name"]
            if not (mirror / V.LOG).exists():
                print(f"  ! vault '{vault['name']}' unreachable: {ex}")
                continue
            print(f"  ! vault '{vault['name']}' unreachable: {ex} — reading its last mirror")
        sources.append((vault["name"], Store(mirror).entries()))

    for prefix in args.hashes:
        matches = [(origin, e, src) for origin, src in sources for e in src
                   if e.kind not in ("keep", "retract") and e.h.startswith(prefix)]
        if not matches:
            raise SystemExit(f"no entry matches hash prefix '{prefix}' "
                             f"(searched local + {len(sources) - 1} vault(s))")
        if len(matches) > 1:
            lines = [f"  {e.h}  [{e.kind}] {e.topic or '—'}  ({origin})"
                     for origin, e, _ in matches]
            raise SystemExit(f"hash prefix '{prefix}' is ambiguous — "
                             f"{len(matches)} matches:\n" + "\n".join(lines))

        origin, e, src = matches[0]
        retracted = {x.target for x in src if x.kind == "retract" and x.target}
        revoked = {x.target for x in src if x.kind == "keep" and x.target}
        by_h = {x.h: x for x in src}

        print(f"\n  {e.at[:19]}  [{e.kind}] {e.topic or '—'} {e.h}  ({origin})")
        print(textwrap.fill(" ".join(e.text.split()), width=100,
                            initial_indent="          ",
                            subsequent_indent="          "))

        if e.kind == "alias" and e.target:
            parent = by_h.get(e.target)
            snippet = " ".join(parent.text.split())[:80] if parent else "?"
            print(f"          alias of {e.target}  \"{snippet}\"")
        if e.topic:
            same = [x for x in src
                    if x.kind not in ("keep", "retract", "alias")
                    and x.h not in retracted and x.topic == e.topic]
            if same and same[-1].h != e.h:
                print(f"          superseded by {same[-1].h} ({same[-1].at[:10]} write)")
        if e.h not in revoked:
            w = max((ww for x in src if x.h not in retracted
                     for th, ww in x.displaces if th == e.h), default=0.0)
            if w:
                print(f"          displaced {w:.2f} by a later write")
        for th, ww in e.displaces:
            target = by_h.get(th)
            snippet = " ".join(target.text.split())[:80] if target else "?"
            print(f"          displaces {th} [{ww:.2f}]  \"{snippet}\"")
        questions = list(e.questions)
        questions += [x.text for x in src
                      if x.kind == "alias" and x.target == e.h
                      and x.text not in questions]
        for q in questions:
            print(f"          question: \"{q}\"")


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

    all_entries = store.entries()
    retracted = store.retracted_hashes(all_entries)
    entries = [e for e in all_entries
               if e.kind not in ("keep", "retract") and e.h not in retracted]
    last = store.latest_by_topic(all_entries)
    revoked = {e.target for e in all_entries if e.kind == "keep" and e.target}
    displaced = {}
    for e in all_entries:
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


def cmd_topics(args) -> None:
    from . import vaults as V
    store = Store(Path(args.store))
    source = "local"
    if args.from_vault:
        match = [v for v in V.load_vaults(store.path) if v["name"] == args.from_vault]
        if not match:
            raise SystemExit(f"no vault named '{args.from_vault}' (see: mnema vault list)")
        store = Store(V.sync_vault(store.path, match[0]))
        source = args.from_vault

    entries = store.entries()
    retracted = store.retracted_hashes(entries)
    live = [e for e in entries
            if e.kind not in ("keep", "retract", "alias") and e.h not in retracted]
    by_topic: dict[str, list[Entry]] = {}
    anonymous = 0
    for e in live:
        if e.topic:
            by_topic.setdefault(e.topic, []).append(e)
        else:
            anonymous += 1

    topical = sum(len(v) for v in by_topic.values())
    print(f"{source}: {len(by_topic)} topics "
          f"({topical} topical writes, {anonymous} anonymous)")
    width = max((len(t) for t in by_topic), default=0)
    for topic in sorted(by_topic):
        writes = by_topic[topic]
        latest = writes[-1]
        tag = f"  ({len(writes)} writes)" if len(writes) > 1 else ""
        print(f"  {topic:<{width}}  {latest.at[:10]}  {latest.h[:8]}{tag}")


def cmd_forget(args) -> None:
    store = Store(Path(args.store))
    targets = list(args.hashes)
    entries = None
    if args.last:
        entries = store.entries()
        retracted = store.retracted_hashes(entries)
        live = [e for e in entries
                if e.kind not in ("keep", "retract") and e.h not in retracted]
        targets.extend(e.h for e in live[-args.last:])
    if not targets:
        raise SystemExit("nothing to forget: pass hash prefixes and/or --last N")
    forgotten = store.forget(targets, entries=entries)
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


def cmd_update(args) -> None:
    """Pull the source this CLI runs from and bounce the warm daemon."""
    import signal
    import subprocess
    import sys
    import time
    from . import serve as S
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").is_dir():
        raise SystemExit(
            f"this mnema is not running from a git checkout ({root}) — reinstall from "
            f"source to make updates one command:\n"
            f"  git clone https://github.com/front-depiction/mnema ~/mnema\n"
            f"  uv tool install --force --editable ~/mnema")
    git = lambda *a: subprocess.run(["git", "-C", str(root), *a], check=True,
                                    capture_output=True, text=True).stdout.strip()
    before = git("rev-parse", "HEAD")
    print(f"pulling {root} ({before[:8]})", flush=True)
    pulled = subprocess.run(["git", "-C", str(root), "pull", "--ff-only"],
                            capture_output=True, text=True)
    if pulled.returncode != 0:
        raise SystemExit(f"git pull failed:\n{pulled.stderr.strip()}\n"
                         f"resolve it in {root}, then run mnema update again")
    after = git("rev-parse", "HEAD")
    if after == before:
        print(f"already up to date ({after[:8]})")
    else:
        n = git("rev-list", "--count", f"{before}..{after}")
        print(f"updated {before[:8]} → {after[:8]} ({n} commits)")
        for line in git("log", "--oneline", f"{before}..{after}").splitlines()[:12]:
            print(f"  {line}")
        changed = git("diff", "--name-only", before, after).splitlines()
        if "pyproject.toml" in changed:
            print("  ! dependencies may have changed — reinstall once:\n"
                  f"    uv tool install --force --editable {root}")
    daemons = S.running_daemons()
    for sock, pid in daemons:
        os.kill(pid, signal.SIGTERM)
        print(f"stopped warm daemon (pid {pid}) — it pinned the old code")
    if daemons:
        time.sleep(1)
        subprocess.Popen([sys.executable, "-m", "mnema.cli", "serve"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        print("restarted warm daemon on the new code")


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
    entries = store.entries()
    st = store.catch_up(quiet=True, entries=entries)
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
    a_entries, b_entries = a.entries(), b.entries()
    a.catch_up(quiet=True, entries=a_entries)
    b.catch_up(quiet=True, entries=b_entries)
    merged = merge(a, b, Path(args.out), a_entries=a_entries, b_entries=b_entries)
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

    p = sub.add_parser("question", help="attach questions an existing memory answers "
                       "(extra addresses; better recall for that memory)")
    p.add_argument("hash", help="hash prefix of the memory")
    p.add_argument("questions", nargs="+", help="one or more questions it answers")
    p.set_defaults(fn=cmd_question)

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
    p.add_argument("--at", help="ISO timestamp stamped on every entry — papers "
                                "should carry their publication date, not the "
                                "file's mtime")
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
    p.add_argument("--except", dest="except_vaults", action="append",
                   metavar="VAULT", default=[],
                   help="leave a named vault out of this ask (repeatable); "
                        "not with --from")
    p.add_argument("--local", action="store_true", help="skip vaults entirely")
    p.add_argument("--scores", action="store_true",
                   help="show raw support and per-hit cosines (diagnostics)")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("show", help="print the full memory behind a printed hash "
                       "(local or any vault); model-free")
    p.add_argument("hashes", nargs="+", metavar="HASH",
                   help="hash prefixes printed by remember/ask/log")
    p.set_defaults(fn=cmd_show)

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

    p = sub.add_parser("topics", help="list topics sorted, no content — prefixes "
                       "read as a tree (local or a vault); model-free")
    p.add_argument("--from", dest="from_vault", metavar="VAULT",
                   help="list a named vault instead of the local store")
    p.set_defaults(fn=cmd_topics)

    p = sub.add_parser("update", help="pull the latest source and restart the warm daemon")
    p.set_defaults(fn=cmd_update)

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
