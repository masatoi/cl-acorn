# Inference Examples Design

**Date**: 2026-03-13
**Topic**: cl-acorn.inference examples (HMC, NUTS, VI)
**Status**: Approved

## Goal

Add three examples under `examples/` that demonstrate the `cl-acorn.inference` package
(HMC, NUTS, VI) using the same Bayesian linear regression model, allowing direct
comparison of the three methods.

## Approach

Three independent, self-contained examples sharing the same statistical model.
Each example lives in its own directory and runs with `(load "examples/NN-.../main.lisp")`.

```
examples/
  09-hmc-bayes-regression/
    README.md
    main.lisp
  10-nuts-bayes-regression/
    README.md
    main.lisp
  11-vi-bayes-regression/
    README.md
    main.lisp
```

## Shared Model

Bayesian linear regression:

```
y_i = w * x_i + b + ε_i,   ε_i ~ N(0, 0.5²)

Priors:
  w ~ N(0, 1)
  b ~ N(0, 10)
  σ = 0.5  (fixed)

log p(w, b | data) ∝
  Σ dist:normal-log-pdf(y_i | w*x_i + b, 0.5)
  + dist:normal-log-pdf(w | 0, 1)
  + dist:normal-log-pdf(b | 0, 10)
```

- Synthetic data: 50 points, true w=2.0, b=1.0, σ=0.5
- Generated inline in each `main.lisp` (no separate `data.lisp`)
- Each example prints posterior mean, std, and accept rate, compared to true values

## Example 09: HMC

**Directory**: `examples/09-hmc-bayes-regression/`
**API**:
```lisp
(infer:hmc log-posterior '(0.0d0 0.0d0)
           :n-samples 1000 :n-warmup 500
           :step-size 0.05d0 :n-leapfrog 10
           :adapt-step-size t)
```
**Output**: posterior mean/std for w and b, accept rate.
**README focus**: What HMC is, fixed trajectory length, when to use.

## Example 10: NUTS

**Directory**: `examples/10-nuts-bayes-regression/`
**API**:
```lisp
(infer:nuts log-posterior '(0.0d0 0.0d0)
            :n-samples 1000 :n-warmup 500
            :adapt-step-size t)
```
**Output**: posterior mean/std for w and b, accept rate.
**README focus**: How NUTS differs from HMC (automatic trajectory length via no-U-turn criterion).

## Example 11: VI (ADVI)

**Directory**: `examples/11-vi-bayes-regression/`
**API**:
```lisp
(infer:vi log-posterior 2
          :n-iterations 2000 :n-elbo-samples 10 :lr 0.01d0)
```
**Output**: variational mean/std for w and b, final ELBO.
**README focus**: Mean-field ADVI vs MCMC; speed vs accuracy trade-off.

## File Structure per Example

Each `main.lisp` contains:
1. `(asdf:load-system :cl-acorn)`
2. `defpackage` + `in-package`
3. Synthetic data generation (fixed seed via `make-random-state`)
4. `log-posterior` function definition
5. Sampler/optimizer call
6. `print-results` helper
7. `(run-example)` called at the end

Each `README.md` contains:
- Background: what the method does (2-3 paragraphs)
- Model specification
- Usage instructions
- Expected output

## Success Criteria

- All three `(load ...)` run without error
- Posterior means are within 2 standard deviations of true values (w=2.0, b=1.0)
- Output is readable and compares estimated vs true values
- Each README explains the method clearly for a reader unfamiliar with that algorithm
