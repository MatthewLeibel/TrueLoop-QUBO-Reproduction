"""
EXPERIMENT 1 -- The feedback-channel law: scalar energy search versus
retained vector regulation, swept in problem size.

Two instance families (sparse MaxCut form, degree ~8):
  planted  known optimum; the local-field vector F = Qs is globally aligned
  sk       frustrated spin glass; local fields are only locally aligned

ARMS at equal evaluation budget (2,400 sample evaluations per cell):
  VECTOR   runtime steered by the per-variable local-field vector F(Q,s),
           O(m) to compute, one vector per round
  SCALAR   identical runtime loop, but fed only elite SAMPLES (the ranking
           channel: log2(16) ~ 4 bits/round, no per-variable vector)
  GREEDY   single-flip greedy charged one evaluation per proposed flip
           (the measurement-priced incumbent)
  BLIND    best of the same number of uniform random samples (floor)
Reference ceiling: free-digital simulated annealing (30 sweeps, unpriced),
the "use a laptop" row; all qualities are excess-over-random ratios
(value - random) / (reference - random).

VERIFICATION BARS (a reproduction must satisfy all four):
  P1 planted, VECTOR >= 0.80 at every n up to 4096 (the solve regime).
  P2 SCALAR monotonically decays with n in both families and is below 0.25
     on planted by n = 2048 (direction starvation).
  P3 sk, VECTOR in [0.5, 0.95] at n >= 512: improves far over blind but
     TRAPS below the free-digital ceiling (the honesty row; if VECTOR
     matched the ceiling on a frustrated spin glass, be suspicious).
  P4 BLIND <= 0.20 at n >= 512; GREEDY >= 0.85 at n = 128 (alive on its
     home ground, so the comparison is not rigged). At larger n the priced
     greedy collapses because 2,400 evaluations is under one sweep of the
     variables: that collapse is itself a finding, reported in the tables.
Checkpoint: results/exp1.json.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qubo_common as Q

SEEDS = tuple(range(1, 11))
NS = (128, 512, 2048, 4096)
ROUNDS = 150                       # x BATCH(16) = 2400 evaluations
EVALS = ROUNDS * Q.BATCH

def build_adj(n, a, b, w):
    adj = [[] for _ in range(n)]
    for k in range(len(a)):
        adj[a[k]].append((b[k], w[k])); adj[b[k]].append((a[k], w[k]))
    return [np.array(x, dtype=object) if x else np.zeros((0, 2)) for x in adj]

def make_flip(x_adj):
    n, adj = x_adj
    def gain(x, i):
        g = 0.0
        for j, wij in adj[i]:
            j = int(j)
            g += wij if x[i] == x[j] else -wij
        return g
    return gain

def cell(kind, n, seed):
    a, b, w, _ = Q.gen_graph(kind, n, seed)
    adj = build_adj(n, a, b, w)
    gain = make_flip((n, adj))
    score = lambda X: Q.cut_batch(X, a, b, w)
    one = lambda x: float(Q.cut_batch(x[None, :], a, b, w)[0])
    rnd = float(np.mean(Q.cut_batch(
        (np.random.default_rng(seed).random((32, n)) < 0.5).astype(np.int8), a, b, w)))
    ref = max(Q.anneal_free(one, gain, n, seed + r, sweeps=30) for r in range(3))
    r_ = lambda v: round((v - rnd) / (ref - rnd + 1e-9), 4)
    fb = lambda x: Q.field_feedback(x, a, b, w, n)
    out = {}
    v, _ = Q.runtime_steer(n, seed, ROUNDS, score, fb, sense=-1.0)
    out["VECTOR"] = r_(v)
    out["SCALAR"] = r_(Q.runtime_scalar(n, seed, ROUNDS, score))
    g, _ = Q.greedy_priced(one, gain, n, seed, EVALS)
    out["GREEDY"] = r_(g)
    out["BLIND"] = r_(Q.blind_best(score, n, seed, EVALS))
    return out

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    path = "results/exp1.json"
    res = {}
    try: res = json.load(open(path))
    except Exception: pass
    for kind in ("planted", "sk"):
        for n in NS:
            seeds = SEEDS if n <= 2048 else SEEDS[:5]
            for seed in seeds:
                key = "%s_n%d_s%d" % (kind, n, seed)
                if key in res: continue
                res[key] = cell(kind, n, seed)
                json.dump(res, open(path, "w"))
            line = "%s n=%4d: " % (kind, n)
            for arm in ("VECTOR", "SCALAR", "GREEDY", "BLIND"):
                vs = [res["%s_n%d_s%d" % (kind, n, s)][arm] for s in seeds]
                line += "%s=%.3f  " % (arm, float(np.mean(vs)))
            print(line, flush=True)
