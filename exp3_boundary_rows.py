"""
EXPERIMENT 3 -- The boundary rows: where the solvability law says NO.

Three negative rows of the regime table, measured:
  dense    arbitrary dense QUBO: the only generic feedback F = Qs costs
           O(n^2) per round AND is only locally aligned; both facts
           reported (quality plus measured feedback cost)
  trap     frustrated sparse spin glass (from Exp 1): vector feedback
           improves far over blind but traps below the free-digital
           ceiling; the gap IS the row
  digital  free-evaluation lane: unpriced simulated annealing at defaults
           is the reference ceiling (ratio 1.0 by construction); every
           priced arm sits below it, so digitally evaluable landscapes
           belong to digital solvers

VERIFICATION BARS:
  P1 dense: VECTOR below 0.85 at n = 2048 (no generic solve claim) and its
     measured per-round feedback cost grows ~quadratically (cost ratio
     n=2048 vs n=512 above 8x).
  P2 trap (from results/exp1.json): sk VECTOR at n = 4096 in [0.5, 0.85],
     strictly below 1.0.
  P3 digital: no priced arm reaches 0.97 of the free ceiling at n >= 2048.
Checkpoint: results/exp3.json.
"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qubo_common as Q

SEEDS = (1, 2, 3, 4, 5)
ROUNDS = 150
EVALS = ROUNDS * Q.BATCH

def gen_dense(n, seed):
    r = np.random.default_rng(seed * 3313 + n)
    M = r.normal(0, 1.0, (n, n)); M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    return M

def cell_dense(n, seed):
    M = gen_dense(n, seed)
    def score(X):
        S = 2.0 * X - 1.0
        return -np.einsum("bi,ij,bj->b", S, M, S) / 2.0   # minimize s'Ms
    one = lambda x: float(score(x[None, :])[0])
    def gain(x, i):
        s = 2.0 * x - 1.0
        return float(2.0 * s[i] * (M[i] @ s))
    rnd = float(np.mean(score(
        (np.random.default_rng(seed).random((32, n)) < 0.5).astype(np.int8))))
    ref = max(Q.anneal_free(one, gain, n, seed + r, sweeps=30) for r in range(3))
    r_ = lambda v: round((v - rnd) / (ref - rnd + 1e-9), 4)
    t0 = time.time()
    def fb(x):
        s = 2.0 * x - 1.0
        return -(M @ s) * 1.0            # F = -Qs: O(n^2) per round
    v, _ = Q.runtime_steer(n, seed, ROUNDS, score, fb, sense=+1.0)
    fb_cost_s = (time.time() - t0)
    g, _ = Q.greedy_priced(one, gain, n, seed, EVALS)
    return {"VECTOR": r_(v), "GREEDY": r_(g),
            "BLIND": r_(Q.blind_best(score, n, seed, EVALS)),
            "steer_wall_s": round(fb_cost_s, 2)}

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    path = "results/exp3.json"
    res = {}
    try: res = json.load(open(path))
    except Exception: pass
    for n in (512, 2048):
        for seed in SEEDS:
            key = "dense_n%d_s%d" % (n, seed)
            if key in res: continue
            res[key] = cell_dense(n, seed)
            json.dump(res, open(path, "w"))
        line = "dense n=%4d: " % n
        for arm in ("VECTOR", "GREEDY", "BLIND"):
            vs = [res["dense_n%d_s%d" % (n, s)][arm] for s in SEEDS]
            line += "%s=%.3f  " % (arm, float(np.mean(vs)))
        ws = [res["dense_n%d_s%d" % (n, s)]["steer_wall_s"] for s in SEEDS]
        line += "steer_wall=%.1fs" % float(np.mean(ws))
        print(line, flush=True)
    e1 = json.load(open("results/exp1.json"))
    sk = [e1["sk_n4096_s%d" % s]["VECTOR"] for s in (1, 2, 3, 4, 5)]
    res["trap_sk4096_vector"] = round(float(np.mean(sk)), 4)
    json.dump(res, open(path, "w"))
    print("trap row (sk n=4096 VECTOR vs free ceiling 1.0): %.3f" % res["trap_sk4096_vector"])
