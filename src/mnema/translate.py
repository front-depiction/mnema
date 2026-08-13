"""Translation layers: anything -> list[Entry]. To support a new source, write
a function that yields Entries; the store neither knows nor cares where
knowledge comes from.

Built-in translators:
  generic_jsonl   {"text": ..., "topic"?: ..., "at"?: ..., "kind"?: ..., "slots"?: {}}
  plan_ledger     tagged JSONL event ledgers ({_tag, at, node?, text}; prose tags only)
"""

from __future__ import annotations

import json
from pathlib import Path

from .store import Entry


def generic_jsonl_lines(lines) -> list[Entry]:
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        d = json.loads(line)
        out.append(Entry(text=d["text"], topic=d.get("topic"),
                         at=d.get("at") or Entry(text="").at,
                         kind=d.get("kind", "note"), slots=d.get("slots", {})))
    return out


def generic_jsonl(path: Path) -> list[Entry]:
    return generic_jsonl_lines(Path(path).read_text().splitlines())


PLAN_LEDGER_TAGS = {"Note", "ForkRuled"}


def plan_ledger(path: Path) -> list[Entry]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        tag = d.get("_tag")
        if tag not in PLAN_LEDGER_TAGS:
            continue
        if tag == "Note":
            text = d["text"]
        else:  # ForkRuled
            text = f"fork {d.get('node', '')} ruled: {d.get('choice', '')}"
        out.append(Entry(text=text, topic=d.get("node"), at=d["at"], kind=tag))
    return out


def markdown(path: Path) -> list[Entry]:
    """One memory per heading-bounded section. The topic is the section's SLOT
    identity — `<path>#<heading-slug>` — so sibling sections never supersede
    each other (disjoint addresses), while re-ingesting an edited file lands
    changed sections on their existing address and supersedes the old version.
    Unchanged files dedupe entirely (timestamp = file mtime, part of the
    entry hash). Run from a stable working directory so paths stay stable."""
    import re
    import time
    p = Path(path)
    at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(p.stat().st_mtime))
    rel = str(p)
    out: list[Entry] = []
    heading, buf = "intro", []

    def flush():
        text = "\n".join(buf).strip()
        if len(text) >= 50:
            slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "intro"
            title = "" if heading == "intro" else f"{heading}\n"
            out.append(Entry(text=f"{title}{text}", topic=f"{rel}#{slug}",
                             at=at, kind="doc"))

    for line in p.read_text().splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush()
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    flush()
    return out


NAV_SLUGS = {"contents", "table-of-contents", "list-of-tables", "list-of-figures", "index"}
PACK_WORDS = 300


def paper(path: Path) -> list[Entry]:
    """Like `markdown`, but for whole papers: each section's paragraphs pack
    into ~300-word chunks so every chunk fits the retrieval horizon. Chunk
    topics extend the section slot — `<path>#<slug>`, `<path>#<slug>/2`, ... —
    disjoint addresses, so nothing inside one paper supersedes anything else,
    while re-ingesting an edited paper lands changed chunks on their existing
    address. Pure-navigation sections (table of contents, index) are dropped.
    Non-text formats (pdf, docx, epub, ...) convert via anydoc first."""
    import re
    import time
    p = Path(path)
    if p.suffix.lower() in (".md", ".markdown", ".txt"):
        md = p.read_text()
    else:
        try:
            import anydoc
        except ImportError:
            raise SystemExit(
                f"reading {p.suffix} needs the optional document converter:\n"
                '  uv pip install firecrawl-anydoc   (or: pip install "mnema[paper]")')
        md = anydoc.to_markdown(str(p))
    at = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(p.stat().st_mtime))
    rel = str(p)
    out: list[Entry] = []
    heading, buf = "intro", []

    def flush():
        slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "intro"
        if slug in NAV_SLUGS:
            return
        paras = [q.strip() for q in re.split(r"\n\s*\n", "\n".join(buf)) if q.strip()]
        chunks, cur, n = [], [], 0
        for para in paras:
            w = len(para.split())
            if cur and n + w > PACK_WORDS:
                chunks.append("\n\n".join(cur))
                cur, n = [], 0
            cur.append(para)
            n += w
        if cur:
            chunks.append("\n\n".join(cur))
        title = "" if heading == "intro" else f"{heading}\n"
        for i, chunk in enumerate(c for c in chunks if len(c) >= 50):
            suffix = "" if i == 0 else f"/{i + 1}"
            out.append(Entry(text=f"{title}{chunk}", topic=f"{rel}#{slug}{suffix}",
                             at=at, kind="doc"))

    for line in md.splitlines():
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush()
            heading, buf = m.group(2).strip(), []
        else:
            buf.append(line)
    flush()
    return out


TRANSLATORS = {"jsonl": generic_jsonl, "ledger": plan_ledger, "markdown": markdown,
               "paper": paper}
