"""Laws of the algebra. Every property the tool advertises is asserted here,
on random data, with no embedding model involved."""

import numpy as np
import pytest

from mnema import core


def unit_rows(n, d, seed):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def test_chunked_fold_equals_sequential():
    D, n = 96, 150
    K, V = unit_rows(n, D, 1), unit_rows(n, D, 2)
    a = core.fold(core.FoldState.empty(D), K, V, beta=1.0, chunk=17)
    b = core.fold_sequential(core.FoldState.empty(D), K, V, beta=1.0)
    assert np.abs(a.S - b.S).max() < 1e-3
    assert np.abs(a.Lam - b.Lam).max() < 1e-3
    assert a.n == b.n == n


def test_orthonormal_keys_give_exact_last_write_wins():
    D = 64
    Q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((D, D)))
    K = Q[:8].astype(np.float32)                       # 8 orthonormal addresses
    first = unit_rows(8, D, 3)
    second = unit_rows(8, D, 4)
    st = core.fold(core.FoldState.empty(D), K, first)
    st = core.fold(st, K, second)                      # supersede every address
    for i in range(8):
        recovered = st.S @ K[i]
        assert np.abs(recovered - second[i]).max() < 1e-4   # zero residue of first


def test_stale_never_wins_within_capacity():
    D, topics = 512, 100
    K = unit_rows(topics, D, 5)
    old, new = unit_rows(topics, D, 6), unit_rows(topics, D, 7)
    st = core.fold(core.FoldState.empty(D), K, old)
    st = core.fold(st, K, new)
    codebook = np.vstack([old, new])
    for i in range(topics):
        scores = codebook @ (st.S @ K[i])
        assert scores[topics + i] > scores[i]          # new beats its stale twin


def test_fold_is_associative_over_segments():
    D, n = 96, 120
    K, V = unit_rows(n, D, 8), unit_rows(n, D, 9)
    whole = core.fold(core.FoldState.empty(D), K, V)
    st = core.FoldState.empty(D)
    cuts = [0, 37, 90, n]                               # uneven segments
    for a, b in zip(cuts, cuts[1:]):
        st = core.fold(st, K[a:b], V[a:b])
    assert np.abs(st.S - whole.S).max() < 1e-3


def test_support_separates_written_from_unwritten():
    D = 256
    K = unit_rows(50, D, 10)
    V = unit_rows(50, D, 11)
    st = core.fold(core.FoldState.empty(D), K, V)
    written = float(np.mean([core.support(st, K[i]) for i in range(50)]))
    fresh = unit_rows(50, D, 12)
    unwritten = float(np.mean([core.support(st, fresh[i]) for i in range(50)]))
    assert written > 5 * unwritten
    assert unwritten == pytest.approx(1.0, abs=0.5)     # isotropic baseline


def test_slot_binding_addresses_are_distinct_and_reproducible():
    D = 512
    k = unit_rows(1, D, 13)[0]
    a1 = core.bind_slots(k, {"team": "platform"}, seed=7)
    a2 = core.bind_slots(k, {"team": "platform"}, seed=7)
    b = core.bind_slots(k, {"team": "ui"}, seed=7)
    assert np.abs(a1 - a2).max() < 1e-6                 # deterministic
    assert abs(float(a1 @ b)) < 0.2                     # different value = far address
    assert abs(float(a1 @ k)) < 0.2                     # slotted != unslotted


def test_spectral_gauge_tracks_live_topics():
    D = 256
    st = core.fold(core.FoldState.empty(D), unit_rows(40, D, 14), unit_rows(40, D, 15))
    g = core.spectral_gauge(st)
    assert 25 < g["live_topics_estimate"] < 55          # ~40, crosstalk tolerated
    assert g["headroom"] > 0.7


def test_beta_below_one_attenuates_instead_of_erasing():
    D = 64
    Q, _ = np.linalg.qr(np.random.default_rng(16).standard_normal((D, D)))
    k = Q[0].astype(np.float32)
    v_old, v_new = Q[1].astype(np.float32), Q[2].astype(np.float32)
    st = core.fold(core.FoldState.empty(D), k[None], v_old[None], beta=0.5)
    st = core.fold(st, k[None], v_new[None], beta=0.5)
    recovered = st.S @ k
    assert float(recovered @ v_old) > 0.05              # old still present
    assert float(recovered @ v_new) > float(recovered @ v_old)   # but losing
