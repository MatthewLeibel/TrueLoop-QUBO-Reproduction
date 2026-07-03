"""
EXPERIMENT 2 -- The valid solve regime: structured QUBOs with their natural
per-variable feedback.

Two application-shaped families (Exp 1's planted family is the third
positive control):
  constraint  penalty QUBO from linear constraints A x = c (set-partition
              style; n/2 constraints of 6 variables; feasible by
              construction). Natural feedback: the per-variable
              backpropagated residual, O(nnz(A)).
  parity      LDPC-like decoding QUBO (n/2 parity checks of 6 bits;
              consistent syndrome). Natural feedback: per-bit violated-check
              counts (syndrome signal), O(m).

Quality metric: violations resolved, q = 1 - E / E_random, so q = 1 is a
perfect (feasible / decoded) assignment and q = 0 is random.

ARMS at equal budget (2,400 sample evaluations): VECTOR (runtime steered by
the family's natural feedback), SCALAR (same loop, elite samples only),
GREEDY (priced single-flip), BLIND.
VERIFICATION BARS:
  P1 VECTOR >= 0.90 on constraint and >= 0.85 on parity at n = 1024 and
     n = 4096 (this is the regime the solvability law names).
  P2 VECTOR beats GREEDY at n = 4096 in both families.
  P3 SCALAR at n = 4096 is below half of VECTOR in both families.
Checkpoint: results/exp2.json.
"""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qubo_common as Q

SEEDS = tuple(range(1, 11))
NS = (1024, 4096)
ROUNDS = 150
EVALS = ROUNDS * Q.BATCH

def var_index(cols, n):
    idx = [[] for _ in range(n)]
    for r, row in enumerate(cols):
        for v in row:
            idx[v].append(r)
    return [np.array(x, dtype=int) for x in idx]

def cell(kind, n, seed):
    if kind == "constraint":
        cols, c, _ = Q.gen_constraint(n, seed)
        energy = lambda X: Q.constraint_energy(X, cols, c)
        marg_fb = lambda p: Q.constraint_marg_feedback(p, cols, c, n)
        vidx = var_index(cols, n)
        def gain(x, i):                      # energy DECREASE when flipping i
            rows = vidx[i]
            if len(rows) == 0: return 0.0
            res = x[cols[rows]].sum(axis=1) - c[rows]
            d = 1 - 2 * int(x[i])
            return float((res ** 2 - (res + d) ** 2).sum())
    else:
        cols, syn, _ = Q.gen_parity(n, seed)
        energy = lambda X: Q.parity_energy(X, cols, syn)
        marg_fb = lambda p: Q.parity_marg_feedback(p, cols, syn, n)
        vidx = var_index(cols, n)
        def gain(x, i):
            rows = vidx[i]
            if len(rows) == 0: return 0.0
            v0 = ((x[cols[rows]].sum(axis=1) % 2) != syn[rows]).sum()
            x[i] ^= 1
            v1 = ((x[cols[rows]].sum(axis=1) % 2) != syn[rows]).sum()
            x[i] ^= 1
            return float(v0 - v1)
    score = lambda X: -energy(X)             # maximize -violations
    one = lambda x: float(-energy(x[None, :])[0])
    rnd = float(np.mean(-energy(
        (np.random.default_rng(seed).random((32, n)) < 0.5).astype(np.int8))))
    q_ = lambda v: round(1.0 - (-v) / (-rnd + 1e-9), 4)   # 1 - E/E_rand
    out = {}
    v = Q.runtime_steer_marginal(n, seed, ROUNDS, score, marg_fb)
    out["VECTOR"] = q_(v)
    out["SCALAR"] = q_(Q.runtime_scalar(n, seed, ROUNDS, score))
    g, _ = Q.greedy_priced(one, gain, n, seed, EVALS)
    out["GREEDY"] = q_(g)
    out["BLIND"] = q_(Q.blind_best(score, n, seed, EVALS))
    return out

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    path = "results/exp2.json"
    res = {}
    try: res = json.load(open(path))
    except Exception: pass
    for kind in ("constraint", "parity"):
        for n in NS:
            seeds = SEEDS if n <= 1024 else SEEDS[:5]
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
