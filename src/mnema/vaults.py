"""Memory vaults: other stores — a colleague's, a team's, an org's — that
contribute to YOUR queries, each addressable by name.

    mnema vault add https://alice-mbp.tail1234.ts.net/mnema --name alice
    mnema ask "what did we decide about retries?"          # local + every vault
    mnema ask --from alice "any ruling on retries?"        # one vault, by name
    mnema remember "..."                                   # ALWAYS local

Multiplex many for read; write only to your own. Writes cannot reach a vault
by construction: vaults are mirrored read-only and `remember` never touches
them.

Sources are an interface. A vault's address is resolved through the SOURCES
registry, keyed by URL scheme — ship your own adapter (S3, git, IPFS, ssh...)
by registering a callable `addr -> Source` where a Source just fetches named
files. Built-ins: http/https/file URLs, and bare filesystem paths.

Serving your store to teammates is trivial because its files are append-only
and its config immutable:

    cd ~/.mnema && python -m http.server 8377             # LAN
    tailscale serve --bg --set-path /mnema ~/.mnema       # tailnet

Vaults are not transitive: a vault's own vaults.json is deliberately ignored.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Callable, Protocol

VAULTS = "vaults.json"
CONFIG = "config.json"
LOG = "log.jsonl"
DERIVED = ["vec_v.f16", "vec_k.f16", "state.npz", "views.npz",
           "pre_k.f16", "pre_v.f16", "log.off"]


# ------------------------------------------------------------------ sources

class Source(Protocol):
    """Anything that can hand over a store's files by name."""

    def fetch(self, filename: str) -> bytes: ...


class UrlSource:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def fetch(self, filename: str) -> bytes:
        with urllib.request.urlopen(f"{self.base}/{filename}", timeout=15) as r:
            return r.read()

    def size(self, filename: str) -> int:
        req = urllib.request.Request(f"{self.base}/{filename}", method="HEAD")
        with urllib.request.urlopen(req, timeout=15) as r:
            return int(r.headers["Content-Length"])


class DirSource:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()

    def fetch(self, filename: str) -> bytes:
        return (self.path / filename).read_bytes()

    def size(self, filename: str) -> int:
        return (self.path / filename).stat().st_size


SOURCES: dict[str, Callable[[str], Source]] = {
    "http": UrlSource,
    "https": UrlSource,
    "file": UrlSource,
}


def source_for(addr: str) -> Source:
    scheme = addr.split("://", 1)[0] if "://" in addr else ""
    if scheme in SOURCES:
        return SOURCES[scheme](addr)
    if not scheme:
        return DirSource(addr)
    raise SystemExit(f"no source adapter for scheme '{scheme}' "
                     f"(registered: {sorted(SOURCES)}); add one to mnema.vaults.SOURCES")


# ------------------------------------------------------------------ registry

def load_vaults(store_path: Path) -> list[dict]:
    p = Path(store_path) / VAULTS
    if not p.exists():
        return []
    return json.loads(p.read_text())["vaults"]


def _save(store_path: Path, vaults: list[dict]) -> None:
    (Path(store_path) / VAULTS).write_text(json.dumps({"vaults": vaults}, indent=2))


def slug_for(addr: str) -> str:
    tail = addr.rstrip("/").rsplit("/", 1)[-1] or "vault"
    return re.sub(r"[^A-Za-z0-9._-]", "-", tail)


def add_vault(store_path: Path, addr: str, name: str | None = None) -> dict:
    vaults = load_vaults(store_path)
    name = name or slug_for(addr)
    if any(v["name"] == name for v in vaults):
        raise SystemExit(f"vault '{name}' already exists (use --name for a distinct one)")
    source_for(addr).fetch(CONFIG)              # fail fast if unreachable
    vault = {"name": name, "addr": addr.rstrip("/") if "://" in addr else addr}
    vaults.append(vault)
    _save(store_path, vaults)
    return vault


def remove_vault(store_path: Path, name: str) -> bool:
    vaults = load_vaults(store_path)
    kept = [v for v in vaults if v["name"] != name]
    if len(kept) == len(vaults):
        return False
    _save(store_path, kept)
    return True


# ------------------------------------------------------------------ syncing

def sync_vault(store_path: Path, vault: dict) -> Path:
    """Mirror a vault's files under <store>/vaults/<name>/. The log is the
    freshness check: byte-identical log => nothing else is fetched."""
    src = source_for(vault["addr"])
    mirror = Path(store_path) / "vaults" / vault["name"]
    mirror.mkdir(parents=True, exist_ok=True)

    cfg = mirror / CONFIG
    if not cfg.exists():
        cfg.write_bytes(src.fetch(CONFIG))      # immutable after init

    # The log is append-only, so an unchanged SIZE means unchanged content —
    # freshness costs one stat/HEAD, never a re-download of the whole log.
    # No size support (custom adapters) falls back to the byte compare.
    mirror_log = mirror / LOG
    if mirror_log.exists() and (mirror / "state.npz").exists():
        try:
            if src.size(LOG) == mirror_log.stat().st_size:
                return mirror
        except Exception:
            pass

    log = src.fetch(LOG)
    cached = mirror_log.read_bytes() if mirror_log.exists() else None
    if log == cached and (mirror / "state.npz").exists():
        return mirror

    (mirror / LOG).write_bytes(log)
    for name in DERIVED:
        try:
            (mirror / name).write_bytes(src.fetch(name))
        except Exception:
            pass    # a lagging or absent derived file is fine: we fold what exists
    return mirror
