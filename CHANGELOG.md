# Changelog

All notable changes to cl-acorn are documented in this file.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-03-15

### BREAKING CHANGES

#### `infer:vi` signature: `n-params` replaced by `initial-params`

The second positional argument to `infer:vi` changed from an integer count
to a list of initial parameter values. This allows VI to start from a
user-supplied initialisation point (matching the HMC/NUTS API) and eliminates
a separate `n-params` argument.

**Before (0.2.x):**
```lisp
(infer:vi log-pdf-fn 3
  :n-iterations 1000 :n-elbo-samples 10 :lr 0.01d0)
```

**After (0.3.0):**
```lisp
(infer:vi log-pdf-fn '(0.0d0 0.0d0 0.0d0)
  :n-iterations 1000 :n-elbo-samples 10 :lr 0.01d0)
```

**Migration:** Replace the integer `n-params` with a list of that many
`0.0d0` values, or supply meaningful starting points drawn from your prior.
The length of the list still determines the number of parameters to infer.

### Added

- **`cl-acorn.diagnostics` package** (`diag:` nickname): new module providing
  multi-chain MCMC execution, convergence diagnostics, and model comparison.
  - `diag:run-chains` — run N independent MCMC chains (NUTS or HMC), compute
    R-hat, bulk-ESS, tail-ESS, and aggregate accept rates / divergence counts.
    New keyword arguments `:n-leapfrog` (HMC-specific) and `:max-tree-depth`
    (NUTS-specific) allow passing sampler-specific parameters through to the
    underlying sampler without affecting the other sampler.
  - `diag:r-hat`, `diag:bulk-ess`, `diag:tail-ess` — Gelman-Rubin R-hat and
    bulk/tail effective sample size from multi-chain sample arrays.
  - `diag:print-convergence-summary` — formatted per-parameter convergence
    table with R-hat and ESS columns.
  - `diag:waic` — Widely Applicable Information Criterion.
  - `diag:loo` — PSIS leave-one-out cross-validation with Pareto-k diagnostics.
  - `diag:print-model-comparison` — formatted model comparison table.
  - `diag:chain-result` struct with accessors for all diagnostic fields.
- **Condition hierarchy** exported from `cl-acorn.inference`:
  - Base condition `infer:acorn-error` (with reader `infer:acorn-error-message`)
    as the root for all library errors; catch it to handle any cl-acorn error.
  - `infer:model-error` — condition for model-definition failures.
  - `infer:inference-error` — condition for sampler failures.
  - Specific errors: `infer:invalid-parameter-error`,
    `infer:log-pdf-domain-error`, `infer:invalid-initial-params-error`.
  - Informational signal (not an error): `infer:non-finite-gradient-error`
    — a plain `condition` (not `error` subtype) signaled via `signal` when
    gradient computation yields non-finite values; use `handler-bind` to observe.
    Reader: `infer:non-finite-gradient-error-message`.
  - Warning: `infer:high-divergence-warning`.
  - Restarts: `infer:use-fallback-params`, `infer:return-empty-samples`,
    `infer:continue-with-warnings`.
- **`infer:inference-diagnostics` struct** returned as the third value from
  `infer:hmc` and `infer:nuts`, and as the fourth value from `infer:vi`.
  Accessors: `infer:diagnostics-accept-rate`, `infer:diagnostics-n-divergences`,
  `infer:diagnostics-final-step-size`, `infer:diagnostics-n-samples`,
  `infer:diagnostics-n-warmup`, `infer:diagnostics-elapsed-seconds`.
  Predicate: `infer:inference-diagnostics-p`.
- **`infer:hmc` restart support**: `infer:use-fallback-params` restart available
  when initial parameters are invalid.
- **`infer:vi` condition support**: replaces internal `assert` calls with
  proper `infer:invalid-parameter-error` conditions; non-finite gradients
  are reported via warnings rather than hard errors.
- **Posterior summary helpers** for hierarchical models:
  `infer:posterior-mean-vec`, `infer:posterior-sd-vec`,
  `infer:print-team-rankings`.
- **Benchmarks**: AD engine, distribution log-pdf, and HMC/NUTS/VI inference
  benchmarks with Python (JAX/PyTorch) comparison scripts.
- **238 tests** (was 170 at 0.2.0 release).

### Fixed

- `diag:run-chains` divergence denominator corrected to
  `n-chains * n-samples` (post-warmup only; warmup samples were
  previously included in the denominator).
- PSIS tail rank assignment corrected; k-hat threshold tightened.
- NUTS multinomial sampling hardened against edge cases in tree building.
- `infer:vi` gradient clipping prevents `g*g` overflow in Adam's second
  moment accumulator.
- Autocorrelation normalisation and lag cap corrected in `bulk-ess`
  computation.
- `copy-chain-result` and `copy-inference-diagnostics` exported from their
  respective packages.

### Changed

- `infer:nuts` adapts step size by default (`:adapt-step-size t`);
  `infer:hmc` still defaults to `:adapt-step-size nil` (fixed step size).
- All public inference APIs now validate inputs with structured conditions
  rather than bare `assert` or `error` calls, enabling condition-based
  recovery via restarts.

## [0.2.0] - 2026-03-13

### Added

- `infer:nuts` — No-U-Turn Sampler with multinomial tree sampling and
  Nesterov dual averaging step-size adaptation.
- `infer:vi` — mean-field ADVI with reparameterization trick and Adam optimizer.
- `infer:dual-averaging-state` and `infer:update-dual-averaging` — Nesterov
  dual averaging for HMC warmup (Hoffman & Gelman 2014).
- `:adapt-step-size` keyword for `infer:hmc` (backward compatible, default `nil`).
- `infer:leapfrog-step` extracted from HMC for reuse by NUTS.

### Changed

- `infer:hmc` returns three values: `(values samples accept-rate diagnostics)`.

## [0.1.0] - 2026-03-12

### Added

- `cl-acorn.ad` — forward-mode (dual numbers) and reverse-mode (tape-based)
  automatic differentiation. Exports: `ad:derivative`, `ad:gradient`,
  `ad:jacobian-vector-product`, `ad:hessian-vector-product`.
- `cl-acorn.distributions` — 6 distributions with AD-transparent `*-log-pdf`
  functions and plain `*-sample` functions:
  `normal`, `gamma`, `beta`, `uniform`, `bernoulli`, `poisson`.
- `cl-acorn.optimizers` — SGD (stateless) and Adam (stateful, `opt:adam-state`).
- `cl-acorn.inference` — HMC sampler with leapfrog integrator.
- Initial test suite (136 tests).
