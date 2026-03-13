# Diagnostics & Model Comparison Design

**Date**: 2026-03-14
**Topic**: cl-acorn convergence diagnostics and model comparison
**Status**: Approved

## Goal

Add a `cl-acorn.diagnostics` package providing multi-chain MCMC execution,
convergence diagnostics (R-hat, ESS), and model comparison (WAIC, PSIS-LOO).

## Architecture

New module `src/diagnostics/` added to the main `cl-acorn` system (not a
separate loadable system), loaded after `inference`.

```
src/
  diagnostics/
    package.lisp           ← cl-acorn.diagnostics (nickname: diag)
    chains.lisp            ← run-chains, chain-result struct
    convergence.lisp       ← r-hat, bulk-ess, tail-ess
    model-comparison.lisp  ← waic, loo, print-model-comparison
tests/
  diagnostics-test.lisp    ← new test suite
```

Dependency order:
```
cl-acorn.ad
  ↑
cl-acorn.distributions
  ↑
cl-acorn.inference
  ↑
cl-acorn.diagnostics
```

## `chain-result` Struct and `run-chains`

```lisp
(defstruct chain-result
  "Aggregated results from a multi-chain MCMC run."
  (samples         nil)
  (n-chains        0   :type (integer 0))
  (n-samples       0   :type (integer 0))   ; per chain, after warmup
  (n-warmup        0   :type (integer 0))
  (r-hat           nil)                     ; list of double-float, per parameter
  (bulk-ess        nil)                     ; list of double-float
  (tail-ess        nil)                     ; list of double-float
  (accept-rates    nil)                     ; list of double-float, per chain
  (n-divergences   0   :type (integer 0))   ; total across all chains
  (elapsed-seconds 0.0d0 :type double-float))

(defun run-chains (log-pdf-fn initial-params
                   &key (n-chains      4)
                        (n-samples  1000)
                        (n-warmup    500)
                        (sampler    :nuts)   ; :nuts or :hmc
                        (step-size  0.1d0)
                        (adapt-step-size t))
  "Run N-CHAINS independent MCMC chains and return a CHAIN-RESULT
with R-hat and ESS automatically computed.

Each chain starts from INITIAL-PARAMS perturbed by small Gaussian noise
to ensure chains explore from different starting points.")
```

Usage:
```lisp
(defvar result
  (diag:run-chains my-log-posterior '(0.0d0 0.0d0)
                   :n-chains 4 :n-samples 1000 :n-warmup 500))

(diag:chain-result-r-hat result)        ; → (1.001 0.999)
(diag:chain-result-bulk-ess result)     ; → (923.4 887.2)
(diag:chain-result-accept-rates result) ; → (0.81 0.79 0.83 0.80)
```

Design notes:
- Starting points: `initial-params` + N(0, 0.1) jitter per chain
- Execution: sequential (parallel execution noted as future extension)
- Warmup samples discarded before R-hat/ESS computation

## Convergence Diagnostics

### R-hat (Gelman-Rubin)

```
W = mean(within-chain variance per parameter)
B = N * variance(chain means per parameter)
V̂ = (N-1)/N * W + B/N
R-hat = sqrt(V̂ / W)
```

Values close to 1.0 indicate convergence. R-hat > 1.1 is a warning sign.

```lisp
(defun r-hat (chains)
  "Compute per-parameter R-hat from CHAINS (list of chains; each chain
is a list of parameter vectors). Returns list of double-float.")
```

### Bulk-ESS and Tail-ESS

Bulk-ESS via autocorrelation:
```
ρ_t = autocorrelation at lag t (summed across chains)
ESS = M*N / (1 + 2 * Σ_t ρ_t)
```

Tail-ESS via quantile indicators:
```
Compute ESS of I(x ≤ Q25) and I(x ≤ Q75) separately → min of the two
```

```lisp
(defun bulk-ess (chains)
  "Per-parameter bulk effective sample size. Returns list of double-float.")

(defun tail-ess (chains)
  "Per-parameter tail ESS (25th/75th percentile indicators).
Reflects reliability of credible intervals.")
```

### `print-convergence-summary`

```lisp
(defun print-convergence-summary (chain-result)
  "Print convergence diagnostics table to *standard-output*.")
```

Output format:
```
Convergence diagnostics  (4 chains x 1000 samples)
====================================================
Param | R-hat  | Bulk-ESS | Tail-ESS | Status
------|--------|----------|----------|--------
  [0] | 1.001  |    923.4 |    887.2 | ok
  [1] | 0.999  |    843.1 |    799.6 | ok
----------------------------------------------------
Total divergences: 0 / 4000
```

Status: `ok` if R-hat < 1.1 and bulk-ESS > 100, else `warn`.

## Model Comparison

### Prerequisite: `log-likelihood-fn`

User provides a per-data-point log-likelihood function:
```lisp
;; (lambda (params data-point) -> double-float)
(defun my-log-lik (params match)
  (dist:poisson-log-pdf (match-goals match) :rate (exp (first params))))
```

### WAIC

```
lppd   = sum_i log( mean_s p(y_i | theta_s) )
p_waic = sum_i var_s( log p(y_i | theta_s) )
WAIC   = -2 * (lppd - p_waic)
```

Lower WAIC = better predictive accuracy.

```lisp
(defun waic (chain-result log-likelihood-fn data)
  "Compute WAIC from posterior samples.
LOG-LIKELIHOOD-FN: (lambda (params data-point) -> double-float)
DATA: sequence of data points
Returns (values waic p-waic lppd).")
```

### PSIS-LOO

Pareto-smoothed importance sampling LOO cross-validation.

Algorithm per data point i:
1. Compute log importance weights: log w_si = -log p(y_i | theta_s)
2. Stabilize by subtracting max
3. Select top M = min(S/5, 3*sqrt(S)) weights as tail
4. Fit generalized Pareto distribution to tail (Zhang-Stephens estimator)
5. Replace tail weights with fitted Pareto quantiles
6. Normalize weights; compute LOO predictive density

Pareto shape k-hat diagnostics:
- k < 0.5:  reliable
- 0.5-0.7:  acceptable
- k >= 0.7: unreliable (that data point has high influence)

```lisp
(defun loo (chain-result log-likelihood-fn data)
  "Compute PSIS-LOO from posterior samples.
Returns (values loo p-loo k-hats) where K-HATS is a list of
per-data-point Pareto shape parameters.")
```

### `print-model-comparison`

```lisp
(defun print-model-comparison (&rest named-results)
  "Print WAIC and LOO comparison table.
NAMED-RESULTS: alternating name/chain-result pairs with log-lik and data:
  (diag:print-model-comparison
     \"model-a\" result-a log-lik-a data
     \"model-b\" result-b log-lik-b data)")
```

Output format:
```
Model comparison
=======================================================
Model          |   WAIC   | p_waic |   LOO    | p_loo
---------------|----------|--------|----------|-------
dixon-coles    |  1823.4  |   41.2 |  1829.1  |  42.3
home-only      |  2104.7  |    3.1 |  2108.2  |   3.2
=======================================================
Lower is better.
```

## Exports

```lisp
;; chains.lisp
#:run-chains
#:chain-result #:chain-result-p
#:chain-result-samples #:chain-result-n-chains
#:chain-result-n-samples #:chain-result-n-warmup
#:chain-result-r-hat #:chain-result-bulk-ess #:chain-result-tail-ess
#:chain-result-accept-rates #:chain-result-n-divergences
#:chain-result-elapsed-seconds

;; convergence.lisp
#:r-hat #:bulk-ess #:tail-ess
#:print-convergence-summary

;; model-comparison.lisp
#:waic #:loo #:print-model-comparison
```

## Testing

New file: `tests/diagnostics-test.lisp`

Test cases:
- R-hat < 1.05 for 4 chains from same distribution
- R-hat > 1.1 for 4 chains started far apart with few samples
- Bulk-ESS > 10% of N*M for near-independent samples
- `run-chains` on 2D standard normal: R-hat < 1.1, ESS > 100
- WAIC lower for correctly specified model vs misspecified model
- LOO k-hats < 0.5 for majority of points under well-specified model

Success criteria:
- All 184 existing tests continue to pass
- All new diagnostics tests pass
- `(diag:run-chains ...)` callable and returns `chain-result`
- `(diag:waic ...)` and `(diag:loo ...)` return plausible values
