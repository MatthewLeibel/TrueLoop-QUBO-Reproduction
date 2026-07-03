# TrueLoop QUBO Reproduction

When can the TrueLoop runtime solve a large QUBO? This repository measures
the answer, regime by regime, with gold-standard baselines at equal budgets.

**The solvability law under test:** the runtime solves a large QUBO exactly
when the problem can be turned from scalar energy search into retained
vector regulation: each round must expose a cheap (subquadratic)
per-variable feedback vector F(Q, s) that stays aligned with the global
solution basin. Scalar-energy-only QUBOs are out of regime; unstructured
spin glasses admit only locally aligned feedback and trap; structured QUBOs
(constraints, parity checks, planted/low-rank structure) expose exactly the
required signal and are the valid solve regime.

The hosted runtime is a black box behind `swc_backend.open_session`; a free
evaluation key from https://trueloopcompute.com reproduces every cell
(all instances here are n <= 4096). The steering adapters in
`qubo_common.py` are ordinary client-side integration code: they compute
each family's textbook feedback signal (local fields; constraint residuals;
Gallager-style syndrome statistics; relaxation-marginal gradients) and hand
the runtime a measurement and a target. No runtime internals appear
anywhere in this repository.

## Results (excess-over-random ratio vs an unpriced simulated-annealing
reference = 1.0; means over seeds; `results/*.json` are the source of truth)

### Experiment 1 -- the feedback channel, swept in size
(`exp1_feedback_channel.py`; 2,400 sample evaluations per cell, 10 seeds
to n=2048, 5 at 4096)

| Family | n | VECTOR (F = local fields) | SCALAR (ranking only) | GREEDY (priced) | BLIND |
|---|---|---|---|---|---|
| planted | 128 | 0.984 | 0.700 | 0.945 | 0.179 |
| planted | 512 | 0.930 | 0.311 | 0.795 | 0.079 |
| planted | 2048 | 0.893 | 0.158 | 0.375 | 0.041 |
| planted | 4096 | **0.878** | 0.109 | 0.247 | 0.025 |
| sk | 128 | 0.935 | 0.844 | 0.851 | 0.327 |
| sk | 512 | 0.830 | 0.603 | 0.870 | 0.168 |
| sk | 2048 | 0.734 | 0.334 | 0.752 | 0.085 |
| sk | 4096 | 0.695 | 0.232 | 0.512 | 0.055 |

The two laws in one table: the scalar channel decays with n in both
families (direction starvation); the vector channel holds on the aligned
family (planted stays at 0.88) and **traps** on the frustrated one
(sk 0.695, far above blind, clearly below the ceiling).

### Experiment 2 -- the valid solve regime
(`exp2_structured_solve.py`; quality = violations resolved, 1 - E/E_random)

| Family (natural feedback) | n | VECTOR | SCALAR | GREEDY (priced) | BLIND |
|---|---|---|---|---|---|
| constraint (residual gradient, O(nnz)) | 1024 | **0.952** | 0.639 | 0.913 | 0.208 |
| constraint | 4096 | **0.954** | 0.354 | 0.669 | 0.107 |
| parity (message signal, O(mk)) | 1024 | **0.854** | 0.302 | 0.801 | 0.156 |
| parity | 4096 | **0.839** | 0.130 | 0.627 | 0.075 |

Structured feedback is scale-stable: VECTOR is flat from 1024 to 4096 while
the priced incumbent degrades and the scalar channel collapses.

### Experiment 3 -- the boundary rows
(`exp3_boundary_rows.py`)

| Row | Measurement | Verdict |
|---|---|---|
| Dense arbitrary QUBO | VECTOR 0.828 (n=512) -> 0.761 (n=2048); feedback wall time 1.0s -> 16.2s (quadratic) | no generic solve, and the only generic feedback costs O(n^2) |
| Spin-glass trap | sk n=4096 VECTOR = 0.695 vs free ceiling 1.0 | improves, may trap: a fast heuristic, not a solver |
| Free-digital lane | no priced arm reaches 0.97 of the unpriced SA ceiling at n >= 2048 | digitally evaluable landscapes belong to digital solvers |

## Verification bars (all pass; each script asserts or prints its own)

Exp 1: planted VECTOR >= 0.80 at every n; SCALAR monotone decay, < 0.25 on
planted by n=2048; sk VECTOR in [0.5, 0.95] at n >= 512 (the trap, which
also certifies the model is not tilted toward us); BLIND <= 0.20 at
n >= 512; GREEDY >= 0.85 at n=128 (alive on its home ground).
Exp 2: VECTOR >= 0.90 (constraint) and >= 0.83 (parity) at both sizes;
VECTOR beats priced GREEDY at n=4096 in both families; SCALAR below half of
VECTOR at n=4096.
Exp 3: dense VECTOR < 0.85 at n=2048 with > 8x feedback-cost growth;
sk trap in [0.5, 0.85]; free ceiling unreached by every priced arm.

## Rigor and fairness

- Budgets are counted in SAMPLE EVALUATIONS, identically for every arm:
  each steering round draws and scores a 16-sample batch (16 units); greedy
  is charged one unit per proposed flip; blind draws the same batches. The
  per-variable feedback vector is a subquadratic digital computation whose
  cost is reported (and, for the dense row, disqualifying).
- Baselines are gold standards: single-flip greedy with exact incremental
  gains; simulated annealing at textbook defaults as the unpriced
  reference ceiling.
- The feedback signals are textbook objects, named in the code: local
  fields Qs, constraint residual gradients, Gallager bit-flip statistics,
  mean-field parity gradients (message-passing-lite), relaxation-marginal
  gradients.
- Honest scope: ratios are against a 30-sweep SA reference, not proven
  optima; the trap row and the dense row are published limits, not
  footnotes. Nothing here claims NP-hard problems are solved in general.

## Reproduce

```bash
pip install -r requirements.txt
export TRUELOOP_KEY=...     # free key: https://trueloopcompute.com
./run_all.sh                # or run the three scripts individually
```
Scripts checkpoint per cell into `results/*.json` and resume on rerun.

## License

MIT (see LICENSE). The TrueLoop runtime itself is proprietary, accessed as
a hosted service (see NOTICE).
