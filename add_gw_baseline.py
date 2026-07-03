import os
import json
import time
import numpy as np
import cvxpy as cp
from qubo_common import load_problem, evaluate_solution

def goemans_williamson(Q, timeout=60):
    """Goemans-Williamson SDP relaxation + random hyperplane rounding."""
    n = Q.shape[0]
    start = time.time()
    
    # SDP relaxation
    X = cp.Variable((n, n), symmetric=True)
    prob = cp.Problem(cp.Maximize(cp.trace(Q @ X)),
                      [cp.diag(X) == 1, X >> -np.eye(n)])
    
    try:
        prob.solve(solver=cp.SCS, verbose=False, max_iters=5000)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            return None, None, "SDP failed"
    except:
        return None, None, "SDP timeout/error"
    
    sdp_time = time.time() - start
    if sdp_time > timeout:
        return None, None, "timeout"
    
    # Random hyperplane rounding (multiple trials)
    best_cut = -np.inf
    X_val = X.value
    if X_val is None:
        return None, None, "no solution"
    
    best_x = None
    for _ in range(100):  # number of rounding trials
        u = np.random.randn(n)
        x = np.sign(X_val @ u)
        cut = x.T @ Q @ x
        if cut > best_cut:
            best_cut = cut
            best_x = x
    
    total_time = time.time() - start
    return best_x, best_cut, total_time

# Example usage wrapper for your problems
def run_gw_on_instance(problem_name, n, seed, timeout=30):
    Q = load_problem(problem_name, n, seed)  # assuming your loader
    x, cut_value, wall_time = goemans_williamson(Q, timeout)
    quality = evaluate_solution(Q, x) if x is not None else 0.0
    return {
        "quality": float(quality),
        "wall_time": float(wall_time),
        "cut_value": float(cut_value) if cut_value is not None else None
    }

print("GW baseline functions ready. Integrate into your experiments.")
