"""
qubo_common.py -- shared machinery for the TrueLoop QUBO reproduction.

Instance families (all reducible to QUBO; sparse edge-list representation):
  planted   MaxCut with a planted bipartition (known optimum): the positive
            control; its local-field vector is globally aligned
  sk        frustrated spin glass (Sherrington-Kirkpatrick-style, sparse,
            degree ~8): local fields decorrelate from the global optimum
  constraint  penalty QUBO from linear constraints A x = b (set-partition
            style): natural feedback = per-variable backpropagated residual
  parity    LDPC-like parity-check decoding QUBO: natural feedback = per-bit
            violated-check counts (syndrome signal)

BUDGET ACCOUNTING (stated in README): every arm is charged in SAMPLE
EVALUATIONS. Each steering round draws and scores a batch of B=16 candidates
(cost 16); blind sampling draws the same batches; greedy/annealing arms are
charged one evaluation per proposed flip. Equal totals throughout. The
per-variable feedback vector F(Q, s) is computed from the single best sample
of the round; its cost is the subquadratic digital cost listed per family
(O(m) here) and is reported, not hidden.

The hosted runtime is a black box behind swc_backend.open_session. The
steering adapter below is ordinary client-side integration code: it samples
from the configuration, computes the family's feedback vector, and hands the
runtime a measurement and a target. No runtime internals appear anywhere.
"""
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from swc_backend import open_session

BATCH = 16

# --------------------------- instance generators --------------------------- #
def gen_graph(kind, n, seed, d=8):
    """planted | sk : sparse degree-d signed graphs (MaxCut form)."""
    r = np.random.default_rng(seed * 9176 + n)
    m = n * d // 2
    a = r.integers(0, n, m); b = (a + 1 + r.integers(0, n - 1, m)) % n
    if kind == "planted":
        side = r.random(n) < 0.5
        w = np.where(side[a] != side[b], 1.0, -1.0) + r.normal(0, 0.1, m)
        opt = side.astype(np.int8)
    else:                                   # sk: frustrated random couplings
        w = r.normal(0, 1.0, m); opt = None
    return a, b, w.astype(float), opt

def gen_constraint(n, seed, rows=None):
    """Penalty QUBO: minimize ||A x - c||^2, A sparse 0/1, c = A x_star.
    Feasible by construction (hidden x_star). rows ~ n/2."""
    r = np.random.default_rng(seed * 5573 + n)
    rows = rows or n // 2
    k = 6                                   # variables per constraint
    cols = np.stack([r.choice(n, k, replace=False) for _ in range(rows)])
    xstar = (r.random(n) < 0.5).astype(np.int8)
    c = xstar[cols].sum(axis=1)
    return cols, c, xstar

def constraint_energy(X, cols, c):
    return ((X[:, cols].sum(axis=2) - c[None, :]) ** 2).sum(axis=1).astype(float)

def constraint_feedback(x, cols, c, n):
    """F_i = sum over constraints containing i of 2*(A x - c): the exact
    per-variable residual gradient, O(nnz(A))."""
    res = x[cols].sum(axis=1) - c
    F = np.zeros(n)
    np.add.at(F, cols.ravel(), np.repeat(2.0 * res, cols.shape[1]))
    return F

def gen_parity(n, seed, checks=None, k=6):
    """LDPC-like decoding QUBO: parity checks over k bits, syndrome from a
    hidden codeword x_star; energy = number of violated checks."""
    r = np.random.default_rng(seed * 7741 + n)
    checks = checks or n // 2
    cols = np.stack([r.choice(n, k, replace=False) for _ in range(checks)])
    xstar = (r.random(n) < 0.5).astype(np.int8)
    syn = xstar[cols].sum(axis=1) % 2
    return cols, syn, xstar

def parity_energy(X, cols, syn):
    return ((X[:, cols].sum(axis=2) % 2) != syn[None, :]).sum(axis=1).astype(float)

def parity_feedback(x, cols, syn, n, deg=None):
    """Gallager-style bit-flip signal: for bit i touching deg_i checks of
    which v_i are violated, the net checks fixed by flipping i is
    2 v_i - deg_i. Returned as a desired push in x-space:
    F_i = (1 - 2 x_i) * (2 v_i - deg_i). O(m)."""
    viol = ((x[cols].sum(axis=1) % 2) != syn).astype(float)
    v = np.zeros(n)
    np.add.at(v, cols.ravel(), np.repeat(viol, cols.shape[1]))
    if deg is None:
        deg = np.zeros(n)
        np.add.at(deg, cols.ravel(), 1.0)
    return (1.0 - 2.0 * x) * (2.0 * v - deg)

# graph (MaxCut) evaluators / feedback
def cut_batch(X, a, b, w):
    return (w[None, :] * (X[:, a] != X[:, b])).sum(axis=1).astype(float)

def field_feedback(x, a, b, w, n):
    """Local-field vector h_i = sum_j w_ij s_j, s = 2x-1. O(m)."""
    s = 2.0 * x - 1.0
    h = np.zeros(n)
    np.add.at(h, a, w * s[b]); np.add.at(h, b, w * s[a])
    return h

# ------------------------------- baselines --------------------------------- #
def blind_best(score_fn, n, seed, evals):
    r = np.random.default_rng(seed * 71)
    best = None
    for _ in range(max(1, evals // BATCH)):
        X = (r.random((BATCH, n)) < 0.5).astype(np.int8)
        v = score_fn(X)
        m = float(v.max()) if best is None else max(best, float(v.max()))
        best = m
    return best

def greedy_priced(score_fn_one, flip_gain, n, seed, evals):
    """Single-flip greedy charged one evaluation per PROPOSED flip."""
    r = np.random.default_rng(seed * 313)
    x = (r.random(n) < 0.5).astype(np.int8)
    cur = score_fn_one(x)
    for t in range(evals):
        i = t % n
        g = flip_gain(x, i)
        if g > 0:
            x[i] ^= 1; cur += g
    return float(cur), x

def anneal_free(score_fn_one, flip_gain, n, seed, sweeps=30):
    """Simulated annealing at defaults, digital-free lane (reference)."""
    r = np.random.default_rng(seed * 977)
    x = (r.random(n) < 0.5).astype(np.int8)
    cur = score_fn_one(x)
    best = cur
    for s in range(sweeps):
        T = max(1e-3, 2.0 * (1 - s / sweeps))
        for i in r.permutation(n):
            g = flip_gain(x, i)
            if g > 0 or r.random() < math.exp(g / T):
                x[i] ^= 1; cur += g; best = max(best, cur)
    return float(best)

# ------------------------- runtime steering adapter ------------------------ #
def runtime_steer(n, seed, rounds, score_fn, feedback_fn, sense=-1.0,
                  key=None):
    """Client-side integration: sample from the configuration's induced
    distribution, score a batch, compute the family feedback vector from the
    round's best sample, and hand the runtime (measurement, target). The
    runtime itself is a hosted black box. sense=-1 anti-aligns (MaxCut /
    minimize violations); charged BATCH evaluations per round."""
    rng = np.random.default_rng(seed * 17)
    s = open_session(n=n, mode="regulation", target=[0.5] * n,
                     key=key or os.environ.get("TRUELOOP_KEY"))
    phi = np.array(s.phi, dtype=float)
    best = -1e18; best_x = None
    for t in range(rounds):
        p = np.clip(0.5 + 0.5 * np.cos(phi), 1e-4, 1 - 1e-4)
        X = (rng.random((BATCH, n)) < p).astype(np.int8)
        v = score_fn(X)
        bi = int(np.argmax(v))
        if float(v[bi]) > best:
            best = float(v[bi]); best_x = X[bi].copy()
        F = feedback_fn(X[bi])
        scale = np.abs(F).mean() + 1e-9
        beta = 0.5 + 3.5 * t / rounds
        y = 1.0 / (1.0 + np.exp(-sense * beta * F / scale))
        phi = np.array(s.step(np.clip(p, 0, 1).tolist(),
                              target=np.clip(y, 0.02, 0.98).tolist()),
                       dtype=float)
    s.end()
    return best, best_x

def runtime_scalar(n, seed, rounds, score_fn, key=None):
    """Scalar-only channel: identical loop, but the runtime is given no
    per-variable vector; the target is built from elite SAMPLES only (the
    best bitstring itself), which carries the log2(BATCH) bits of the
    ranking and nothing else."""
    rng = np.random.default_rng(seed * 19)
    s = open_session(n=n, mode="regulation", target=[0.5] * n,
                     key=key or os.environ.get("TRUELOOP_KEY"))
    phi = np.array(s.phi, dtype=float)
    best = -1e18
    y = np.full(n, 0.5)
    for t in range(rounds):
        p = np.clip(0.5 + 0.5 * np.cos(phi), 1e-4, 1 - 1e-4)
        X = (rng.random((BATCH, n)) < p).astype(np.int8)
        v = score_fn(X)
        bi = int(np.argmax(v)); best = max(best, float(v[bi]))
        y = 0.8 * y + 0.2 * X[bi]                    # elite-sample target
        phi = np.array(s.step(np.clip(p, 0, 1).tolist(),
                              target=np.clip(y, 0.02, 0.98).tolist()),
                       dtype=float)
    s.end()
    return best


def runtime_steer_marginal(n, seed, rounds, score_fn, marg_feedback_fn,
                           key=None):
    """Marginal-fed steering: the feedback vector is computed from the
    configuration's induced marginals p (relaxation-marginal / message
    signals, the table's structured-feedback class), not from a single
    sample. Target is an incremental nudge y = clip(p + eta_t * F / scale),
    with eta ramped, so the distribution follows the relaxation gradient.
    A batch is still sampled and scored each round (charged) so best-found
    is tracked under the same accounting as every other arm."""
    rng = np.random.default_rng(seed * 17)
    s = open_session(n=n, mode="regulation", target=[0.5] * n,
                     key=key or os.environ.get("TRUELOOP_KEY"))
    phi = np.array(s.phi, dtype=float)
    best = -1e18
    tilt = rng.uniform(-0.06, 0.06, n)              # symmetry breaking
    for t in range(rounds):
        p = np.clip(0.5 + 0.5 * np.cos(phi), 1e-4, 1 - 1e-4)
        if t == 0:
            p = np.clip(p + tilt, 0.02, 0.98)
        X = (rng.random((BATCH - 1, n)) < p).astype(np.int8)
        Xr = np.vstack([X, (p > 0.5).astype(np.int8)])   # include rounding
        v = score_fn(Xr)
        best = max(best, float(v.max()))
        F = marg_feedback_fn(p)
        scale = np.abs(F).max() + 1e-9
        eta = 0.08 + 0.42 * t / rounds
        y = np.clip(p + eta * F / scale, 0.02, 0.98)
        phi = np.array(s.step(np.clip(p, 0, 1).tolist(),
                              target=y.tolist()), dtype=float)
    s.end()
    return best

def constraint_marg_feedback(p, cols, c, n):
    """Exact expected gradient of -||A p - c||^2: F = -2 A^T (A p - c).
    O(nnz(A)); the relaxation-marginal signal."""
    res = p[cols].sum(axis=1) - c
    F = np.zeros(n)
    np.add.at(F, cols.ravel(), np.repeat(-2.0 * res, cols.shape[1]))
    return F

def parity_marg_feedback(p, cols, syn, n):
    """Mean-field parity satisfaction gradient (message-passing-lite):
    with g = 1 - 2p, check r's soft satisfaction is t_r * prod_j g_j,
    t_r = (-1)^{syn_r}; F_i = d/dp_i sum_r satisfaction
        = sum_{r ni i} -2 t_r prod_{j in r, j != i} g_j.  O(m k)."""
    g = 1.0 - 2.0 * np.clip(p, 0.02, 0.98)
    gs = np.where(np.abs(g) < 0.05, np.sign(g + 1e-12) * 0.05, g)
    G = gs[cols]                                    # (checks, k)
    P = G.prod(axis=1)
    t = np.where(syn % 2 == 1, -1.0, 1.0)
    contrib = -2.0 * t[:, None] * (P[:, None] / G)   # leave-one-out
    F = np.zeros(n)
    np.add.at(F, cols.ravel(), contrib.ravel())
    return F
