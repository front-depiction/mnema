"""Daemon protocol: client transparently uses a running server; identical
vectors to direct embedding; silent fallback when absent."""

import tempfile
import threading

import numpy as np

from mnema.serve import EmbedServer, socket_path, try_daemon

from test_store import FakeEmbedder


def test_daemon_roundtrip_and_fallback(monkeypatch):
    # a SHORT dir: macOS caps AF_UNIX paths at ~104 chars (pytest tmp_path is deeper)
    short = tempfile.mkdtemp(prefix="mnema-sock-")
    monkeypatch.setenv("MNEMA_SOCKET_DIR", short)

    assert try_daemon("fake", "passages", ["x"]) is None      # no daemon: fallback

    server = EmbedServer(socket_path("fake"), FakeEmbedder())
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        via = try_daemon("fake", "passages", ["hello", "world"])
        direct = FakeEmbedder().passages(["hello", "world"])
        assert via is not None
        np.testing.assert_allclose(via, direct, atol=1e-6)

        via_q = try_daemon("fake", "queries", ["hello"])
        assert via_q is not None and via_q.shape == (1, 32)
    finally:
        server.shutdown()
