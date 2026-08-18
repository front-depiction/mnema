"""Warm-model daemon. Loading embedding weights dominates CLI latency
(~seconds per invocation); `mnema serve` pays it once and serves embeddings
over a unix socket. The CLI uses a running daemon transparently and falls
back to in-process loading when none is up — no configuration, no flags.

    mnema serve &        # one per model; every store on that model benefits

Protocol: one JSON line per request  {"op": "passages"|"queries", "texts": [...]}
one JSON line per response           {"vectors": [[...], ...]}  or  {"error": "..."}
Sockets live per-model under $MNEMA_SOCKET_DIR (default ~/.cache/mnema)."""

from __future__ import annotations

import json
import os
import re
import socket
import socketserver
import threading
from pathlib import Path

import numpy as np

CONNECT_TIMEOUT = 0.25    # daemon detection must be near-free when absent
REQUEST_TIMEOUT = 600.0   # bulk ingests embed thousands of texts in one call


def socket_path(model_name: str) -> Path:
    root = Path(os.environ.get("MNEMA_SOCKET_DIR",
                               Path.home() / ".cache" / "mnema"))
    root.mkdir(parents=True, exist_ok=True)
    return root / (re.sub(r"[^A-Za-z0-9._-]", "-", model_name) + ".sock")


# ------------------------------------------------------------------ client

def try_daemon(model_name: str, op: str, texts: list[str]) -> np.ndarray | None:
    """One round trip to a running daemon, or None if there isn't one."""
    path = socket_path(model_name)
    if not path.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(CONNECT_TIMEOUT)
            s.connect(str(path))
            s.settimeout(REQUEST_TIMEOUT)
            f = s.makefile("rwb")
            f.write(json.dumps({"op": op, "texts": texts}).encode() + b"\n")
            f.flush()
            resp = json.loads(f.readline())
        if "error" in resp:
            return None
        return np.asarray(resp["vectors"], dtype=np.float32)
    except (OSError, ValueError):
        return None            # stale socket / dead daemon: fall back silently


# ------------------------------------------------------------------ server

class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            req = json.loads(self.rfile.readline())
            emb = self.server.embedder
            with self.server.lock:              # torch encode is not thread-safe
                X = (emb.queries(req["texts"]) if req["op"] == "queries"
                     else emb.passages(req["texts"]))
            out = {"vectors": X.tolist()}
        except Exception as ex:
            out = {"error": f"{type(ex).__name__}: {ex}"}
        self.wfile.write(json.dumps(out).encode() + b"\n")


class EmbedServer(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, path: Path, embedder):
        path.unlink(missing_ok=True)
        super().__init__(str(path), _Handler)
        self.embedder = embedder
        self.lock = threading.Lock()


def pid_path(sock: Path) -> Path:
    return sock.with_suffix(".pid")


def running_daemons() -> list[tuple[Path, int]]:
    """(socket, pid) of every daemon whose pid file names a live process."""
    root = socket_path("probe").parent
    out = []
    for pid_file in root.glob("*.pid"):
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
        except (ValueError, OSError):
            pid_file.unlink(missing_ok=True)
            continue
        out.append((pid_file.with_suffix(".sock"), pid))
    return out


def serve(embedder, path: Path | None = None) -> None:
    import signal
    import sys
    path = path or socket_path(embedder.model_name)
    pid_path(path).write_text(str(os.getpid()))  # findable while still loading
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # so cleanup runs
    try:
        _ = embedder.model                      # load weights before accepting
        server = EmbedServer(path, embedder)
        print(f"mnema daemon: {embedder.model_name} warm on {path}", flush=True)
        server.serve_forever()
    finally:
        path.unlink(missing_ok=True)
        pid_path(path).unlink(missing_ok=True)
