"""The algebra: lifting, key construction, the delta-rule fold, confidence, gauge.

Everything here is pure numpy over unit vectors. The state is a pair:

    S   (D x D)  delta-rule memory      S <- S(I - beta k k^T) + beta v k^T
    Lam (D x D)  key second moment      Lam <- Lam + k k^T   (confidence only)

S learns the function meaning -> current content (supersession = subtraction).
Lam records where in meaning space anything was ever written (order-free by
design: "how densely was this region written" legitimately ignores order).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

CHUNK = 64


# ------------------------------------------------------------------ lifting

def lift_projection(d_embed: int, dim: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((d_embed, dim)) / np.sqrt(dim)).astype(np.float32)


def lift(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Project embeddings to fold space and re-normalize rows."""
    V = X.astype(np.float32) @ R
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return V / n


def rff_projection(d_embed: int, dim: int, sigma: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Random Fourier features: keys through phi(x) = cos(xW + b) approximate a
    Gaussian kernel of bandwidth sigma — the locality dial for addressing."""
    rng = np.random.default_rng(seed + 1)
    W = (rng.standard_normal((d_embed, dim)) / sigma).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, dim).astype(np.float32)
    return W, b


def rff(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    F = np.cos(X.astype(np.float32) @ W + b)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return F / n


# ------------------------------------------------------------------ slots

def _seeded_unit(tag: str, dim: int, seed: int) -> np.ndarray:
    h = int.from_bytes(hashlib.sha256(f"{seed}:{tag}".encode()).digest()[:8], "big")
    v = np.random.default_rng(h).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def bind_slots(k: np.ndarray, slots: dict[str, str], seed: int) -> np.ndarray:
    """Bind role=value structure into a key by circular convolution. Querying
    with the same slots reproduces the address; slot-free keys are untouched."""
    if not slots:
        return k
    dim = k.shape[-1]
    f = np.fft.rfft(k)
    for name in sorted(slots):
        f = f * np.fft.rfft(_seeded_unit(f"{name}={slots[name]}", dim, seed))
    out = np.fft.irfft(f, n=dim).astype(np.float32)
    return out / np.linalg.norm(out)


# ------------------------------------------------------------------ the fold

@dataclass
class FoldState:
    S: np.ndarray     # (D, D) float32
    Lam: np.ndarray   # (D, D) float32
    ksum: np.ndarray  # (D,) running sum of keys — for anisotropy centering
    n: int            # entries folded so far

    @classmethod
    def empty(cls, dim: int) -> "FoldState":
        return cls(np.zeros((dim, dim), np.float32), np.zeros((dim, dim), np.float32),
                   np.zeros(dim, np.float32), 0)


def fold(state: FoldState, K: np.ndarray, V: np.ndarray, beta: float = 1.0,
         chunk: int = CHUNK) -> FoldState:
    """Advance the state through (K, V) rows in order, chunk-parallel.

    Chunked delta rule (UT transform): with G = K_c K_c^T,
        (I + beta * tril(G, -1)) U = beta * (V_c - K_c S^T),   S += U^T K_c.
    Exactly equivalent to the sequential rule; two GEMMs + a c x c solve.
    """
    S, Lam = state.S, state.Lam
    for a in range(0, K.shape[0], chunk):
        Kc = K[a:a + chunk].astype(np.float32)
        Vc = V[a:a + chunk].astype(np.float32)
        c = Kc.shape[0]
        G = Kc @ Kc.T
        A = np.eye(c, dtype=np.float32) + beta * np.tril(G, -1)
        U = np.linalg.solve(A, beta * (Vc - Kc @ S.T))
        S += U.T @ Kc
        Lam += Kc.T @ Kc
        state.ksum += Kc.sum(axis=0)
    return FoldState(S, Lam, state.ksum, state.n + K.shape[0])


def fold_sequential(state: FoldState, K: np.ndarray, V: np.ndarray,
                    beta: float = 1.0) -> FoldState:
    """Reference implementation; used to verify the chunked fold."""
    S, Lam, ksum = state.S.copy(), state.Lam.copy(), state.ksum.copy()
    for i in range(K.shape[0]):
        k, v = K[i].astype(np.float32), V[i].astype(np.float32)
        S += beta * np.outer(v - S @ k, k)
        Lam += np.outer(k, k)
        ksum += k
    return FoldState(S, Lam, ksum, state.n + K.shape[0])


# ------------------------------------------------------------------ reading

def retrieve(state: FoldState, q: np.ndarray) -> np.ndarray:
    """One matvec: the relevance-weighted blend of current content at every
    address the query touches."""
    return state.S @ q


def support(state: FoldState, q: np.ndarray) -> float:
    """Written-ness of the query's region: the CENTERED key second moment
    Lam_c = Lam - n mu mu^T evaluated at q, as a ratio over its isotropic
    baseline tr(Lam_c)/D. Centering removes the common anisotropy cone all
    text embeddings share — without it, every query looks 'written'.
    ~1 means unwritten ground; >> 1 means settled ground. Order-free."""
    if state.n == 0:
        return 0.0
    mu = state.ksum / state.n
    raw = float(q @ state.Lam @ q) - state.n * float(q @ mu) ** 2
    tr_c = float(np.trace(state.Lam)) - state.n * float(mu @ mu)
    if tr_c <= 0.0:
        return 0.0
    baseline = tr_c / state.Lam.shape[0]
    return max(0.0, raw / baseline)


def spectral_gauge(state: FoldState, probes: int = 64, seed: int = 0) -> dict:
    """Capacity self-awareness from the singular spectrum of S, estimated by a
    randomized sketch (a few GEMMs). Effective rank ~ live topics in store;
    headroom = 1 - eff_rank / D. Alarm before the cliff, not after."""
    D = state.S.shape[0]
    rng = np.random.default_rng(seed)
    Y = state.S @ rng.standard_normal((D, min(probes * 4, D))).astype(np.float32)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ state.S
    sv = np.linalg.svd(B, compute_uv=False)
    total, sq = float(sv.sum()), float((sv ** 2).sum())
    eff_rank_sketch = (total * total / sq) if sq > 0 else 0.0
    # Participation ratio of the CENTERED second moment (see support()) —
    # uncentered, the common embedding cone's giant eigenvalue swallows it.
    mu = state.ksum / max(state.n, 1)
    tr = float(np.trace(state.Lam)) - state.n * float(mu @ mu)
    Lmu = state.Lam @ mu
    fro2 = (float((state.Lam * state.Lam).sum())
            - 2.0 * state.n * float(mu @ Lmu)
            + state.n ** 2 * float(mu @ mu) ** 2)
    live_est = (tr * tr / fro2) if fro2 > 0 else 0.0
    return {
        "top_singular_values": [round(float(x), 3) for x in sv[:8]],
        "effective_rank_sketch": round(eff_rank_sketch, 1),
        "live_topics_estimate": round(live_est, 1),
        "dimension": D,
        "headroom": round(max(0.0, 1.0 - live_est / D), 3),
    }
